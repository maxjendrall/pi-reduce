import { buildSessionContext, calculateContextTokens, DynamicBorder, estimateTokens, getSettingsListTheme } from "@mariozechner/pi-coding-agent";
import type { ExtensionAPI, ExtensionCommandContext } from "@mariozechner/pi-coding-agent";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, ImageContent, Message, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage } from "@mariozechner/pi-ai";
import { Container, SettingsList, Text, type SettingItem } from "@mariozechner/pi-tui";

type DistillPreset = "chat" | "reasoning" | "tools" | "no-tools";
type ToolFilterMode = "all" | "include" | "exclude";
type TruncationMode = "head" | "tail";

type DistillCategory =
	| "user"
	| "assistant_thinking"
	| "assistant_comment"
	| "assistant_final"
	| "assistant_status"
	| "tool_call"
	| "tool_result"
	| "bash_execution"
	| "custom_message"
	| "branch_summary"
	| "compaction_summary";

interface DistillConfig {
	includeUser: boolean;
	includeAssistantThinking: boolean;
	includeAssistantComment: boolean;
	includeAssistantFinal: boolean;
	includeAssistantStatus: boolean;
	includeToolCalls: boolean;
	includeToolResults: boolean;
	includeBashExecution: boolean;
	includeCustomMessages: boolean;
	includeBranchSummaries: boolean;
	includeCompactionSummaries: boolean;
	toolFilterMode: ToolFilterMode;
	toolNames: string[];
	toolCallMaxChars?: number;
	toolCallMaxLines?: number;
	toolCallMaxApproxTokens?: number;
	toolResultMaxChars?: number;
	toolResultMaxLines?: number;
	toolResultMaxApproxTokens?: number;
	toolResultTruncation: TruncationMode;
}

interface DistillState {
	lastConfig: DistillConfig;
	lastPreset?: DistillPreset;
}

interface DistillStats {
	sourceMessages: number;
	keptMessages: number;
	droppedMessages: number;
	truncatedToolCalls: number;
	truncatedToolResults: number;
	sourceApproxTokens: number;
	keptApproxTokens: number;
	sourceHistoricalUsage: UsageBreakdown;
	keptHistoricalUsage: UsageBreakdown;
	sourceMessageBreakdown: MessageBreakdown;
	keptMessageBreakdown: MessageBreakdown;
	sourceApproxByCategory: Record<DistillCategory, number>;
	keptApproxByCategory: Record<DistillCategory, number>;
	keptByCategory: Record<DistillCategory, number>;
}

interface UsageBreakdown {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	total: number;
	assistantMessages: number;
}

interface MessageBreakdown {
	user: number;
	assistant: number;
	toolCalls: number;
	toolResults: number;
	total: number;
}

interface CategoryMeta {
	key: DistillCategory;
	label: string;
	get(config: DistillConfig): boolean;
	set(config: DistillConfig, enabled: boolean): void;
}

interface LocalCustomMessage<T = unknown> {
	role: "custom";
	customType: string;
	content: string | (TextContent | ImageContent)[];
	display: boolean;
	details?: T;
	timestamp: number;
}

interface BranchSummaryMessage {
	role: "branchSummary";
	summary: string;
	fromId: string;
	timestamp: number;
}

interface CompactionSummaryMessage {
	role: "compactionSummary";
	summary: string;
	tokensBefore: number;
	timestamp: number;
}

interface BashExecutionMessage {
	role: "bashExecution";
	command: string;
	output: string;
	exitCode: number | undefined;
	cancelled: boolean;
	truncated: boolean;
	fullOutputPath?: string;
	excludeFromContext?: boolean;
	timestamp: number;
}

type DistillMessage = AgentMessage | LocalCustomMessage | BranchSummaryMessage | CompactionSummaryMessage | BashExecutionMessage;

const STATE_ENTRY_TYPE = "distill-config";

const ZERO_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const DEFAULT_CONFIG: DistillConfig = {
	includeUser: true,
	includeAssistantThinking: false,
	includeAssistantComment: false,
	includeAssistantFinal: true,
	includeAssistantStatus: false,
	includeToolCalls: false,
	includeToolResults: false,
	includeBashExecution: false,
	includeCustomMessages: false,
	includeBranchSummaries: false,
	includeCompactionSummaries: false,
	toolFilterMode: "all",
	toolNames: [],
	toolResultTruncation: "head",
};

const PRESETS: Record<DistillPreset, { description: string; config: DistillConfig }> = {
	chat: {
		description: "user + assistant final",
		config: {
			...DEFAULT_CONFIG,
			includeUser: true,
			includeAssistantFinal: true,
		},
	},
	reasoning: {
		description: "user + thinking + comments + final",
		config: {
			...DEFAULT_CONFIG,
			includeUser: true,
			includeAssistantThinking: true,
			includeAssistantComment: true,
			includeAssistantFinal: true,
		},
	},
	tools: {
		description: "user + tool calls + tool results",
		config: {
			...DEFAULT_CONFIG,
			includeUser: true,
			includeToolCalls: true,
			includeToolResults: true,
		},
	},
	"no-tools": {
		description: "user + assistant comments + final (no tool trace)",
		config: {
			...DEFAULT_CONFIG,
			includeUser: true,
			includeAssistantComment: true,
			includeAssistantFinal: true,
		},
	},
};

const DISTILL_CATEGORIES: DistillCategory[] = [
	"user",
	"assistant_thinking",
	"assistant_comment",
	"assistant_final",
	"assistant_status",
	"tool_call",
	"tool_result",
	"bash_execution",
	"custom_message",
	"branch_summary",
	"compaction_summary",
];

const CATEGORY_META: CategoryMeta[] = [
	{
		key: "user",
		label: "User messages",
		get: (config) => config.includeUser,
		set: (config, enabled) => {
			config.includeUser = enabled;
		},
	},
	{
		key: "assistant_thinking",
		label: "Assistant thinking",
		get: (config) => config.includeAssistantThinking,
		set: (config, enabled) => {
			config.includeAssistantThinking = enabled;
		},
	},
	{
		key: "assistant_comment",
		label: "Assistant comments (tool-using turns)",
		get: (config) => config.includeAssistantComment,
		set: (config, enabled) => {
			config.includeAssistantComment = enabled;
		},
	},
	{
		key: "assistant_final",
		label: "Assistant final messages",
		get: (config) => config.includeAssistantFinal,
		set: (config, enabled) => {
			config.includeAssistantFinal = enabled;
		},
	},
	{
		key: "assistant_status",
		label: "Assistant status / aborted / errors",
		get: (config) => config.includeAssistantStatus,
		set: (config, enabled) => {
			config.includeAssistantStatus = enabled;
		},
	},
	{
		key: "tool_call",
		label: "Tool calls",
		get: (config) => config.includeToolCalls,
		set: (config, enabled) => {
			config.includeToolCalls = enabled;
		},
	},
	{
		key: "tool_result",
		label: "Tool results",
		get: (config) => config.includeToolResults,
		set: (config, enabled) => {
			config.includeToolResults = enabled;
		},
	},
	{
		key: "bash_execution",
		label: "User bash executions (! / !!)",
		get: (config) => config.includeBashExecution,
		set: (config, enabled) => {
			config.includeBashExecution = enabled;
		},
	},
	{
		key: "custom_message",
		label: "Custom / extension messages",
		get: (config) => config.includeCustomMessages,
		set: (config, enabled) => {
			config.includeCustomMessages = enabled;
		},
	},
	{
		key: "branch_summary",
		label: "Existing branch summaries",
		get: (config) => config.includeBranchSummaries,
		set: (config, enabled) => {
			config.includeBranchSummaries = enabled;
		},
	},
	{
		key: "compaction_summary",
		label: "Existing compaction summaries",
		get: (config) => config.includeCompactionSummaries,
		set: (config, enabled) => {
			config.includeCompactionSummaries = enabled;
		},
	},
];

function cloneConfig(config: DistillConfig): DistillConfig {
	return {
		...config,
		toolNames: [...config.toolNames],
	};
}

function normalizeText(text: string): string {
	return text.replace(/\r\n/g, "\n");
}

function trimText(text: string): string {
	return normalizeText(text).trim();
}

function safeJson(value: unknown): string {
	try {
		return JSON.stringify(value ?? {}, null, 2) ?? "{}";
	} catch {
		return String(value ?? "{}");
	}
}

function parsePositiveInt(value: string | undefined): number | undefined {
	if (!value) return undefined;
	const trimmed = value.trim();
	if (!trimmed) return undefined;
	const parsed = Number.parseInt(trimmed, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function normalizeToolNames(input: string): string[] {
	return Array.from(
		new Set(
			input
				.split(/[\s,]+/)
				.map((part) => part.trim())
				.filter(Boolean),
		),
	);
}

function sanitizeConfig(input: Partial<DistillConfig> | undefined): DistillConfig {
	const config = cloneConfig(DEFAULT_CONFIG);
	if (!input) return config;

	for (const key of [
		"includeUser",
		"includeAssistantThinking",
		"includeAssistantComment",
		"includeAssistantFinal",
		"includeAssistantStatus",
		"includeToolCalls",
		"includeToolResults",
		"includeBashExecution",
		"includeCustomMessages",
		"includeBranchSummaries",
		"includeCompactionSummaries",
	] as const) {
		if (typeof input[key] === "boolean") config[key] = input[key] as DistillConfig[typeof key];
	}

	if (input.toolFilterMode === "all" || input.toolFilterMode === "include" || input.toolFilterMode === "exclude") {
		config.toolFilterMode = input.toolFilterMode;
	}
	config.toolNames = Array.isArray(input.toolNames)
		? normalizeToolNames(input.toolNames.join(","))
		: [...DEFAULT_CONFIG.toolNames];

	for (const key of [
		"toolCallMaxChars",
		"toolCallMaxLines",
		"toolCallMaxApproxTokens",
		"toolResultMaxChars",
		"toolResultMaxLines",
		"toolResultMaxApproxTokens",
	] as const) {
		const value = input[key];
		if (typeof value === "number" && Number.isFinite(value) && value > 0) {
			config[key] = Math.floor(value);
		} else {
			delete config[key];
		}
	}

	config.toolResultTruncation = input.toolResultTruncation === "tail" ? "tail" : "head";
	return config;
}

function isUserMessage(message: DistillMessage): message is UserMessage {
	return message.role === "user";
}

function isAssistantMessage(message: DistillMessage): message is AssistantMessage {
	return message.role === "assistant";
}

function isToolResultMessage(message: DistillMessage): message is ToolResultMessage {
	return message.role === "toolResult";
}

function isCustomMessage(message: DistillMessage): message is LocalCustomMessage {
	return message.role === "custom";
}

function isBashExecutionMessage(message: DistillMessage): message is BashExecutionMessage {
	return message.role === "bashExecution";
}

function isBranchSummaryMessage(message: DistillMessage): message is BranchSummaryMessage {
	return message.role === "branchSummary";
}

function isCompactionSummaryMessage(message: DistillMessage): message is CompactionSummaryMessage {
	return message.role === "compactionSummary";
}

function toolAllowed(toolName: string, config: DistillConfig): boolean {
	if (config.toolFilterMode === "all") return true;
	const names = new Set(config.toolNames.map((name) => name.toLowerCase()));
	const normalized = toolName.toLowerCase();
	if (config.toolFilterMode === "include") return names.has(normalized);
	return !names.has(normalized);
}

function hasStatusInfo(message: AssistantMessage): boolean {
	return message.stopReason !== "stop" && message.stopReason !== "toolUse";
}

function makeStatusText(message: AssistantMessage): string {
	const base =
		message.stopReason === "aborted"
			? "Assistant output was aborted."
			: message.stopReason === "error"
				? "Assistant turn ended with an error."
				: `Assistant stop reason: ${message.stopReason}.`;
	return message.errorMessage ? `${base}\n\n${trimText(message.errorMessage)}` : base;
}

function parsePreset(args: string): DistillPreset | "advanced" | "last" | undefined {
	const token = args.trim().toLowerCase();
	if (!token) return undefined;
	if (token === "advanced" || token === "custom") return "advanced";
	if (token === "last") return "last";
	if (token === "chat" || token === "reasoning" || token === "tools" || token === "no-tools") return token;
	return undefined;
}

function effectiveMaxChars(maxChars: number | undefined, maxApproxTokens: number | undefined): number | undefined {
	const tokenChars = maxApproxTokens ? maxApproxTokens * 4 : undefined;
	if (maxChars && tokenChars) return Math.min(maxChars, tokenChars);
	return maxChars ?? tokenChars;
}

function truncateText(
	text: string,
	options: {
		maxChars?: number;
		maxLines?: number;
		maxApproxTokens?: number;
		mode: TruncationMode;
		noteLabel: string;
	},
): { text: string; truncated: boolean } {
	const normalized = normalizeText(text);
	const maxChars = effectiveMaxChars(options.maxChars, options.maxApproxTokens);
	const maxLines = options.maxLines;
	let result = normalized;
	let truncated = false;

	if (maxLines && result.split("\n").length > maxLines) {
		const lines = result.split("\n");
		const kept = options.mode === "tail" ? lines.slice(-maxLines) : lines.slice(0, maxLines);
		const omitted = lines.length - kept.length;
		result = options.mode === "tail"
			? `[reduce truncated ${options.noteLabel}: kept last ${kept.length}/${lines.length} lines, omitted ${omitted}]\n${kept.join("\n")}`
			: `${kept.join("\n")}\n[reduce truncated ${options.noteLabel}: kept first ${kept.length}/${lines.length} lines, omitted ${omitted}]`;
		truncated = true;
	}

	if (maxChars && result.length > maxChars) {
		const originalLength = result.length;
		if (options.mode === "tail") {
			result = `[reduce truncated ${options.noteLabel}: kept last ${maxChars}/${originalLength} chars]\n${result.slice(-maxChars)}`;
		} else {
			result = `${result.slice(0, maxChars)}\n[reduce truncated ${options.noteLabel}: kept first ${maxChars}/${originalLength} chars]`;
		}
		truncated = true;
	}

	return { text: result, truncated };
}

function truncateToolArguments(toolCall: ToolCall, config: DistillConfig): { toolCall: ToolCall; truncated: boolean } {
	if (!config.toolCallMaxChars && !config.toolCallMaxLines && !config.toolCallMaxApproxTokens) {
		return { toolCall, truncated: false };
	}

	const serialized = safeJson(toolCall.arguments ?? {});
	const truncated = truncateText(serialized, {
		maxChars: config.toolCallMaxChars,
		maxLines: config.toolCallMaxLines,
		maxApproxTokens: config.toolCallMaxApproxTokens,
		mode: "head",
		noteLabel: `tool args for ${toolCall.name}`,
	});

	if (!truncated.truncated) {
		return { toolCall, truncated: false };
	}

	return {
		toolCall: {
			...toolCall,
			arguments: {
				__reduced: true,
				__truncated: true,
				preview: truncated.text,
			},
		},
		truncated: true,
	};
}

function truncateToolResultContent(
	message: ToolResultMessage,
	config: DistillConfig,
): { content: (TextContent | ImageContent)[]; truncated: boolean } {
	if (!config.toolResultMaxChars && !config.toolResultMaxLines && !config.toolResultMaxApproxTokens) {
		return { content: [...message.content], truncated: false };
	}

	const images = message.content.filter((part): part is ImageContent => part.type === "image");
	const text = message.content
		.filter((part): part is TextContent => part.type === "text")
		.map((part) => part.text)
		.join("\n\n");

	if (!text.trim()) {
		return { content: [...images], truncated: false };
	}

	const truncated = truncateText(text, {
		maxChars: config.toolResultMaxChars,
		maxLines: config.toolResultMaxLines,
		maxApproxTokens: config.toolResultMaxApproxTokens,
		mode: config.toolResultTruncation,
		noteLabel: `tool result for ${message.toolName}`,
	});

	return {
		content: [{ type: "text", text: truncated.text }, ...images],
		truncated: truncated.truncated,
	};
}

function truncateBashOutput(message: BashExecutionMessage, config: DistillConfig): { output: string; truncated: boolean } {
	if (!config.toolResultMaxChars && !config.toolResultMaxLines && !config.toolResultMaxApproxTokens) {
		return { output: message.output, truncated: false };
	}
	const truncated = truncateText(message.output, {
		maxChars: config.toolResultMaxChars,
		maxLines: config.toolResultMaxLines,
		maxApproxTokens: config.toolResultMaxApproxTokens,
		mode: config.toolResultTruncation,
		noteLabel: `bash output for ${message.command}`,
	});
	return { output: truncated.text, truncated: truncated.truncated };
}

function createStats(): DistillStats {
	return {
		sourceMessages: 0,
		keptMessages: 0,
		droppedMessages: 0,
		truncatedToolCalls: 0,
		truncatedToolResults: 0,
		sourceApproxTokens: 0,
		keptApproxTokens: 0,
		sourceHistoricalUsage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0, assistantMessages: 0 },
		keptHistoricalUsage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0, assistantMessages: 0 },
		sourceMessageBreakdown: { user: 0, assistant: 0, toolCalls: 0, toolResults: 0, total: 0 },
		keptMessageBreakdown: { user: 0, assistant: 0, toolCalls: 0, toolResults: 0, total: 0 },
		sourceApproxByCategory: createApproxTokenBreakdown(),
		keptApproxByCategory: createApproxTokenBreakdown(),
		keptByCategory: {
			user: 0,
			assistant_thinking: 0,
			assistant_comment: 0,
			assistant_final: 0,
			assistant_status: 0,
			tool_call: 0,
			tool_result: 0,
			bash_execution: 0,
			custom_message: 0,
			branch_summary: 0,
			compaction_summary: 0,
		},
	};
}

function createMessageBreakdown(): MessageBreakdown {
	return { user: 0, assistant: 0, toolCalls: 0, toolResults: 0, total: 0 };
}

function createApproxTokenBreakdown(): Record<DistillCategory, number> {
	return {
		user: 0,
		assistant_thinking: 0,
		assistant_comment: 0,
		assistant_final: 0,
		assistant_status: 0,
		tool_call: 0,
		tool_result: 0,
		bash_execution: 0,
		custom_message: 0,
		branch_summary: 0,
		compaction_summary: 0,
	};
}

function estimateApproxContextTokens(messages: DistillMessage[]): number {
	return messages.reduce((sum, message) => sum + estimateTokens(message as AgentMessage), 0);
}

function estimateTextTokens(text: string): number {
	return Math.ceil(text.length / 4);
}

function estimateToolCallTokens(toolCall: ToolCall): number {
	return Math.ceil((toolCall.name.length + JSON.stringify(toolCall.arguments ?? {}).length) / 4);
}

function sumApproxBreakdown(breakdown: Record<DistillCategory, number>): number {
	return DISTILL_CATEGORIES.reduce((sum, category) => sum + (breakdown[category] ?? 0), 0);
}

function sumHistoricalUsage(messages: DistillMessage[]): UsageBreakdown {
	const usage: UsageBreakdown = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0, assistantMessages: 0 };
	for (const message of messages) {
		if (!isAssistantMessage(message)) continue;
		usage.assistantMessages += 1;
		usage.input += message.usage.input ?? 0;
		usage.output += message.usage.output ?? 0;
		usage.cacheRead += message.usage.cacheRead ?? 0;
		usage.cacheWrite += message.usage.cacheWrite ?? 0;
		usage.total += calculateContextTokens(message.usage);
	}
	return usage;
}

function computeMessageBreakdown(messages: DistillMessage[]): MessageBreakdown {
	const breakdown = createMessageBreakdown();
	breakdown.total = messages.length;

	for (const message of messages) {
		if (isUserMessage(message)) {
			breakdown.user += 1;
			continue;
		}
		if (isAssistantMessage(message)) {
			breakdown.assistant += 1;
			breakdown.toolCalls += message.content.filter((part) => part.type === "toolCall").length;
			continue;
		}
		if (isToolResultMessage(message)) {
			breakdown.toolResults += 1;
		}
	}

	return breakdown;
}

function computeApproxCategoryTokens(messages: DistillMessage[]): Record<DistillCategory, number> {
	const breakdown = createApproxTokenBreakdown();

	for (const message of messages) {
		if (isUserMessage(message)) {
			breakdown.user += estimateTokens(message);
			continue;
		}

		if (isAssistantMessage(message)) {
			const hasToolCall = message.content.some((part) => part.type === "toolCall");
			for (const part of message.content) {
				if (part.type === "thinking") {
					if (trimText(part.thinking)) breakdown.assistant_thinking += estimateTextTokens(part.thinking);
					continue;
				}
				if (part.type === "text") {
					if (!trimText(part.text)) continue;
					const category: DistillCategory = hasToolCall ? "assistant_comment" : "assistant_final";
					breakdown[category] += estimateTextTokens(part.text);
					continue;
				}
				if (part.type === "toolCall") {
					breakdown.tool_call += estimateToolCallTokens(part);
				}
			}

			if (hasStatusInfo(message)) {
				const statusText = trimText(makeStatusText(message));
				if (statusText) breakdown.assistant_status += estimateTextTokens(statusText);
			}
			continue;
		}

		if (isToolResultMessage(message)) {
			breakdown.tool_result += estimateTokens(message);
			continue;
		}

		if (isBashExecutionMessage(message)) {
			breakdown.bash_execution += estimateTokens(message as AgentMessage);
			continue;
		}

		if (isCustomMessage(message)) {
			breakdown.custom_message += estimateTokens(message as AgentMessage);
			continue;
		}

		if (isBranchSummaryMessage(message)) {
			breakdown.branch_summary += estimateTextTokens(message.summary);
			continue;
		}

		if (isCompactionSummaryMessage(message)) {
			breakdown.compaction_summary += estimateTextTokens(message.summary);
		}
	}

	return breakdown;
}

function formatInt(value: number): string {
	return new Intl.NumberFormat("en-US").format(Math.max(0, Math.round(value)));
}

function formatPercent(value: number): string {
	return `${value.toFixed(1)}%`;
}

function reductionBar(percent: number, width = 20): string {
	const clamped = Math.max(0, Math.min(100, percent));
	const filled = Math.round((clamped / 100) * width);
	return `${"█".repeat(filled)}${"░".repeat(Math.max(0, width - filled))}`;
}

function reductionPercent(before: number, after: number): number {
	if (before <= 0) return 0;
	return Math.max(0, ((before - after) / before) * 100);
}

function summaryValueLine(label: string, after: number, before: number, options?: { approx?: boolean }): string {
	const prefix = options?.approx ? "~" : "";
	return `${label}: ${prefix}${formatInt(after)} (Before: ${prefix}${formatInt(before)}, ${formatPercent(reductionPercent(before, after))} reduction)`;
}

function computeSourceCategoryCounts(messages: DistillMessage[]): Record<DistillCategory, number> {
	const counts = createStats().keptByCategory;
	for (const message of messages) {
		if (isUserMessage(message)) {
			counts.user += 1;
			continue;
		}
		if (isAssistantMessage(message)) {
			const hasToolCall = message.content.some((part) => part.type === "toolCall");
			const hasThinking = message.content.some((part) => part.type === "thinking" && trimText(part.thinking));
			const textCount = message.content.filter((part) => part.type === "text" && trimText(part.text)).length;
			if (hasThinking) counts.assistant_thinking += 1;
			if (textCount > 0) {
				if (hasToolCall) counts.assistant_comment += 1;
				else counts.assistant_final += 1;
			}
			if (hasStatusInfo(message)) counts.assistant_status += 1;
			if (hasToolCall) counts.tool_call += 1;
			continue;
		}
		if (isToolResultMessage(message)) {
			counts.tool_result += 1;
			continue;
		}
		if (isBashExecutionMessage(message)) {
			counts.bash_execution += 1;
			continue;
		}
		if (isCustomMessage(message)) {
			counts.custom_message += 1;
			continue;
		}
		if (isBranchSummaryMessage(message)) {
			counts.branch_summary += 1;
			continue;
		}
		if (isCompactionSummaryMessage(message)) {
			counts.compaction_summary += 1;
		}
	}
	return counts;
}

function distillMessages(messages: DistillMessage[], config: DistillConfig): { messages: Array<Message | LocalCustomMessage | BashExecutionMessage>; stats: DistillStats } {
	const distilled: Array<Message | LocalCustomMessage | BashExecutionMessage> = [];
	const stats = createStats();
	stats.sourceMessages = messages.length;
	stats.sourceHistoricalUsage = sumHistoricalUsage(messages);
	stats.sourceMessageBreakdown = computeMessageBreakdown(messages);
	stats.sourceApproxByCategory = computeApproxCategoryTokens(messages);
	stats.sourceApproxTokens = sumApproxBreakdown(stats.sourceApproxByCategory);

	for (const message of messages) {
		if (isUserMessage(message)) {
			if (config.includeUser) {
				distilled.push({ ...message, content: Array.isArray(message.content) ? [...message.content] : message.content });
				stats.keptMessages += 1;
				stats.keptByCategory.user += 1;
			} else {
				stats.droppedMessages += 1;
			}
			continue;
		}

		if (isAssistantMessage(message)) {
			const hasToolCall = message.content.some((part) => part.type === "toolCall");
			const selectedContent: AssistantMessage["content"] = [];
			let categoryCounted = {
				assistant_thinking: false,
				assistant_comment: false,
				assistant_final: false,
				assistant_status: false,
				tool_call: false,
			};

			for (const part of message.content) {
				if (part.type === "thinking") {
					if (!config.includeAssistantThinking || !trimText(part.thinking)) continue;
					selectedContent.push({ ...part });
					categoryCounted.assistant_thinking = true;
					continue;
				}

				if (part.type === "text") {
					if (!trimText(part.text)) continue;
					if (hasToolCall) {
						if (!config.includeAssistantComment) continue;
						categoryCounted.assistant_comment = true;
					} else {
						if (!config.includeAssistantFinal) continue;
						categoryCounted.assistant_final = true;
					}
					selectedContent.push({ ...part });
					continue;
				}

				if (part.type === "toolCall") {
					if (!config.includeToolCalls || !toolAllowed(part.name, config)) continue;
					const truncated = truncateToolArguments(part, config);
					selectedContent.push(truncated.toolCall);
					categoryCounted.tool_call = true;
					if (truncated.truncated) stats.truncatedToolCalls += 1;
				}
			}

			if (config.includeAssistantStatus && hasStatusInfo(message)) {
				const statusText = trimText(makeStatusText(message));
				if (statusText) {
					selectedContent.push({ type: "text", text: statusText });
					categoryCounted.assistant_status = true;
				}
			}

			if (selectedContent.length === 0) {
				stats.droppedMessages += 1;
				continue;
			}

			const distilledAssistant: AssistantMessage = {
				...message,
				content: selectedContent,
				usage: ZERO_USAGE,
				stopReason:
					selectedContent.some((part) => part.type === "toolCall")
						? "toolUse"
						: config.includeAssistantStatus && hasStatusInfo(message)
							? message.stopReason
							: "stop",
				errorMessage: config.includeAssistantStatus ? message.errorMessage : undefined,
			};

			distilled.push(distilledAssistant);
			stats.keptMessages += 1;
			for (const key of Object.keys(categoryCounted) as Array<keyof typeof categoryCounted>) {
				if (categoryCounted[key]) stats.keptByCategory[key] += 1;
			}
			continue;
		}

		if (isToolResultMessage(message)) {
			if (!config.includeToolResults || !toolAllowed(message.toolName, config)) {
				stats.droppedMessages += 1;
				continue;
			}
			const truncated = truncateToolResultContent(message, config);
			const distilledToolResult: ToolResultMessage = {
				...message,
				content: truncated.content,
				details: {
					__reduced: true,
					toolName: message.toolName,
					truncated: truncated.truncated,
					originalDetailsOmitted: message.details !== undefined,
				},
			};
			distilled.push(distilledToolResult);
			stats.keptMessages += 1;
			stats.keptByCategory.tool_result += 1;
			if (truncated.truncated) stats.truncatedToolResults += 1;
			continue;
		}

		if (isBashExecutionMessage(message)) {
			if (!config.includeBashExecution) {
				stats.droppedMessages += 1;
				continue;
			}
			const truncated = truncateBashOutput(message, config);
			distilled.push({ ...message, output: truncated.output, truncated: message.truncated || truncated.truncated });
			stats.keptMessages += 1;
			stats.keptByCategory.bash_execution += 1;
			if (truncated.truncated) stats.truncatedToolResults += 1;
			continue;
		}

		if (isCustomMessage(message)) {
			if (!config.includeCustomMessages) {
				stats.droppedMessages += 1;
				continue;
			}
			distilled.push({ ...message, details: { __reduced: true, originalDetailsOmitted: message.details !== undefined } });
			stats.keptMessages += 1;
			stats.keptByCategory.custom_message += 1;
			continue;
		}

		if (isBranchSummaryMessage(message)) {
			if (!config.includeBranchSummaries) {
				stats.droppedMessages += 1;
				continue;
			}
			const customMessage: LocalCustomMessage = {
				role: "custom",
				customType: "reduce-branch-summary",
				content: `Source branch summary:\n\n${message.summary}`,
				display: true,
				details: { sourceRole: "branchSummary", fromId: message.fromId },
				timestamp: message.timestamp,
			};
			distilled.push(customMessage);
			stats.keptMessages += 1;
			stats.keptByCategory.branch_summary += 1;
			continue;
		}

		if (isCompactionSummaryMessage(message)) {
			if (!config.includeCompactionSummaries) {
				stats.droppedMessages += 1;
				continue;
			}
			const customMessage: LocalCustomMessage = {
				role: "custom",
				customType: "reduce-compaction-summary",
				content: `Source compaction summary:\n\n${message.summary}`,
				display: true,
				details: { sourceRole: "compactionSummary", tokensBefore: message.tokensBefore },
				timestamp: message.timestamp,
			};
			distilled.push(customMessage);
			stats.keptMessages += 1;
			stats.keptByCategory.compaction_summary += 1;
			continue;
		}

		stats.droppedMessages += 1;
	}

	stats.keptApproxByCategory = computeApproxCategoryTokens(distilled as DistillMessage[]);
	stats.keptApproxTokens = sumApproxBreakdown(stats.keptApproxByCategory);
	stats.keptHistoricalUsage = sumHistoricalUsage(distilled as DistillMessage[]);
	stats.keptMessageBreakdown = computeMessageBreakdown(distilled as DistillMessage[]);

	return { messages: distilled, stats };
}

function configSummary(config: DistillConfig): string[] {
	const enabledCategories = CATEGORY_META.filter((meta) => meta.get(config)).map((meta) => meta.label);
	const lines = [
		`Categories: ${enabledCategories.length > 0 ? enabledCategories.join(", ") : "none"}`,
		`Tool filter: ${
			config.toolFilterMode === "all"
				? "all tools"
				: `${config.toolFilterMode} ${config.toolNames.length > 0 ? config.toolNames.join(", ") : "(none specified)"}`
		}`,
		`Tool call budget: ${[
			config.toolCallMaxApproxTokens ? `~${config.toolCallMaxApproxTokens} tok` : undefined,
			config.toolCallMaxChars ? `${config.toolCallMaxChars} chars` : undefined,
			config.toolCallMaxLines ? `${config.toolCallMaxLines} lines` : undefined,
		]
			.filter(Boolean)
			.join(", ") || "off"}`,
		`Tool result budget: ${[
			config.toolResultMaxApproxTokens ? `~${config.toolResultMaxApproxTokens} tok` : undefined,
			config.toolResultMaxChars ? `${config.toolResultMaxChars} chars` : undefined,
			config.toolResultMaxLines ? `${config.toolResultMaxLines} lines` : undefined,
		]
			.filter(Boolean)
			.join(", ") || "off"} (${config.toolResultTruncation})`,
	];
	return lines;
}

function statsSummary(stats: DistillStats): string[] {
	const breakdownLines = DISTILL_CATEGORIES.filter(
		(category) => (stats.sourceApproxByCategory[category] ?? 0) > 0 || (stats.keptApproxByCategory[category] ?? 0) > 0,
	).map((category) => {
		const source = stats.sourceApproxByCategory[category] ?? 0;
		const kept = stats.keptApproxByCategory[category] ?? 0;
		const label = CATEGORY_META.find((meta) => meta.key === category)?.label ?? category;
		return `  ${summaryValueLine(label, kept, source, { approx: true })}`;
	});

	return [
		"Tokens:",
		`  ${summaryValueLine("Approx Context", stats.keptApproxTokens, stats.sourceApproxTokens, { approx: true })}`,
		"",
		"  Recorded Usage (source session):",
		`    Input: ${formatInt(stats.sourceHistoricalUsage.input)}`,
		`    Output: ${formatInt(stats.sourceHistoricalUsage.output)}`,
		`    Cache Read: ${formatInt(stats.sourceHistoricalUsage.cacheRead)}`,
		`    Cache Write: ${formatInt(stats.sourceHistoricalUsage.cacheWrite)}`,
		`    Total: ${formatInt(stats.sourceHistoricalUsage.total)}`,
		"",
		"  Recorded Usage (new reduced session):",
		`    Input: ${formatInt(stats.keptHistoricalUsage.input)}`,
		`    Output: ${formatInt(stats.keptHistoricalUsage.output)}`,
		`    Cache Read: ${formatInt(stats.keptHistoricalUsage.cacheRead)}`,
		`    Cache Write: ${formatInt(stats.keptHistoricalUsage.cacheWrite)}`,
		`    Total: ${formatInt(stats.keptHistoricalUsage.total)}`,
		"    Note: copied messages do not preserve historical usage tags.",
		"",
		"Messages:",
		`  ${summaryValueLine("User", stats.keptMessageBreakdown.user, stats.sourceMessageBreakdown.user)}`,
		`  ${summaryValueLine("Assistant", stats.keptMessageBreakdown.assistant, stats.sourceMessageBreakdown.assistant)}`,
		`  ${summaryValueLine("Tool Calls", stats.keptMessageBreakdown.toolCalls, stats.sourceMessageBreakdown.toolCalls)}`,
		`  ${summaryValueLine("Tool Results", stats.keptMessageBreakdown.toolResults, stats.sourceMessageBreakdown.toolResults)}`,
		`  ${summaryValueLine("Total", stats.keptMessageBreakdown.total, stats.sourceMessageBreakdown.total)}`,
		"",
		"Blocks (approx tokens):",
		...breakdownLines,
		"",
		"Truncation:",
		`  Tool Calls: ${formatInt(stats.truncatedToolCalls)}`,
		`  Tool Results: ${formatInt(stats.truncatedToolResults)}`,
	];
}

function compactFinalSummary(sessionName: string, stats: DistillStats): string {
	const approxReduction = reductionPercent(stats.sourceApproxTokens, stats.keptApproxTokens);
	return [
		`Reduced → ${sessionName}`,
		`ctx ${reductionBar(approxReduction)} ${formatPercent(approxReduction)} reduced`,
		`~${formatInt(stats.sourceApproxTokens)} → ~${formatInt(stats.keptApproxTokens)} tokens · msgs ${formatInt(stats.keptMessageBreakdown.total)}/${formatInt(stats.sourceMessageBreakdown.total)} kept`,
		`blocks think ~${formatInt(stats.sourceApproxByCategory.assistant_thinking)}→~${formatInt(stats.keptApproxByCategory.assistant_thinking)} · calls ~${formatInt(stats.sourceApproxByCategory.tool_call)}→~${formatInt(stats.keptApproxByCategory.tool_call)} · results ~${formatInt(stats.sourceApproxByCategory.tool_result)}→~${formatInt(stats.keptApproxByCategory.tool_result)}`,
		`usage(src) in ${formatInt(stats.sourceHistoricalUsage.input)} · out ${formatInt(stats.sourceHistoricalUsage.output)} · cache ${formatInt(stats.sourceHistoricalUsage.cacheRead + stats.sourceHistoricalUsage.cacheWrite)} · total ${formatInt(stats.sourceHistoricalUsage.total)}`,
	].join("\n");
}

function restoreState(ctx: ExtensionCommandContext): DistillState {
	let state: DistillState = { lastConfig: cloneConfig(DEFAULT_CONFIG) };
	for (const entry of ctx.sessionManager.getBranch()) {
		if (entry.type !== "custom" || entry.customType !== STATE_ENTRY_TYPE || typeof entry.data !== "object" || !entry.data) {
			continue;
		}
		const data = entry.data as { lastConfig?: Partial<DistillConfig>; lastPreset?: DistillPreset };
		state = {
			lastConfig: sanitizeConfig(data.lastConfig),
			lastPreset: data.lastPreset,
		};
	}
	return state;
}

async function pickPreset(ctx: ExtensionCommandContext): Promise<DistillPreset | "advanced" | undefined> {
	const options = [
		`chat — ${PRESETS.chat.description}`,
		`reasoning — ${PRESETS.reasoning.description}`,
		`tools — ${PRESETS.tools.description}`,
		`no-tools — ${PRESETS["no-tools"].description}`,
		"advanced — open the full reduce wizard",
	];
	const selection = await ctx.ui.select("Reduce preset", options);
	if (!selection) return undefined;
	if (selection.startsWith("advanced")) return "advanced";
	return selection.split(" — ")[0] as DistillPreset;
}

async function configureCategories(
	ctx: ExtensionCommandContext,
	config: DistillConfig,
	counts: Record<DistillCategory, number>,
): Promise<DistillConfig | undefined> {
	return ctx.ui.custom<DistillConfig>((tui, theme, _kb, done) => {
		const working = cloneConfig(config);
		const items: SettingItem[] = CATEGORY_META.map((meta) => ({
			id: meta.key,
			label: `${meta.label} (${counts[meta.key] ?? 0})`,
			currentValue: meta.get(working) ? "include" : "exclude",
			values: ["include", "exclude"],
		}));

		const container = new Container();
		container.addChild(new DynamicBorder((s: string) => theme.fg("accent", s)));
		container.addChild(new Text(theme.fg("accent", theme.bold("Reduce categories")), 1, 0));
		container.addChild(new Text(theme.fg("muted", "Toggle which message types survive into the new session."), 1, 0));

		const settings = new SettingsList(
			items,
			Math.min(items.length + 2, 18),
			getSettingsListTheme(),
			(id, value) => {
				const meta = CATEGORY_META.find((item) => item.key === id);
				if (!meta) return;
				meta.set(working, value === "include");
			},
			() => done(cloneConfig(working)),
			{ enableSearch: true },
		);
		container.addChild(settings);
		container.addChild(new Text(theme.fg("dim", "Use arrow keys to toggle. Type / to search. Close to continue."), 1, 0));
		container.addChild(new DynamicBorder((s: string) => theme.fg("accent", s)));

		return {
			render: (width: number) => container.render(width),
			invalidate: () => container.invalidate(),
			handleInput: (data: string) => {
				settings.handleInput?.(data);
				tui.requestRender();
			},
		};
	});
}

async function configureToolFilter(ctx: ExtensionCommandContext, config: DistillConfig): Promise<DistillConfig | undefined> {
	const working = cloneConfig(config);
	const modeSelection = await ctx.ui.select("Tool filter", [
		"all — keep all tools",
		"include — keep only the listed tools",
		"exclude — drop the listed tools",
	]);
	if (!modeSelection) return undefined;
	working.toolFilterMode = modeSelection.startsWith("include") ? "include" : modeSelection.startsWith("exclude") ? "exclude" : "all";
	if (working.toolFilterMode !== "all") {
		const names = await ctx.ui.input(
			working.toolFilterMode === "include" ? "Include only these tools" : "Exclude these tools",
			working.toolNames.join(", "),
		);
		if (names === undefined) return undefined;
		working.toolNames = normalizeToolNames(names);
	} else {
		working.toolNames = [];
	}
	return working;
}

async function configureBudgets(ctx: ExtensionCommandContext, config: DistillConfig): Promise<DistillConfig | undefined> {
	const working = cloneConfig(config);
	const mode = await ctx.ui.select("Tool result truncation direction", [
		"head — keep the start of tool output",
		"tail — keep the end of tool output",
	]);
	if (!mode) return undefined;
	working.toolResultTruncation = mode.startsWith("tail") ? "tail" : "head";

	const prompts: Array<{
		title: string;
		placeholder: string;
		value: number | undefined;
		assign: (value: number | undefined) => void;
	}> = [
		{
			title: "Tool call approx-token budget (blank = off)",
			placeholder: working.toolCallMaxApproxTokens ? String(working.toolCallMaxApproxTokens) : "",
			value: working.toolCallMaxApproxTokens,
			assign: (value) => {
				working.toolCallMaxApproxTokens = value;
			},
		},
		{
			title: "Tool call char budget (blank = off)",
			placeholder: working.toolCallMaxChars ? String(working.toolCallMaxChars) : "",
			value: working.toolCallMaxChars,
			assign: (value) => {
				working.toolCallMaxChars = value;
			},
		},
		{
			title: "Tool call line budget (blank = off)",
			placeholder: working.toolCallMaxLines ? String(working.toolCallMaxLines) : "",
			value: working.toolCallMaxLines,
			assign: (value) => {
				working.toolCallMaxLines = value;
			},
		},
		{
			title: "Tool result approx-token budget (blank = off)",
			placeholder: working.toolResultMaxApproxTokens ? String(working.toolResultMaxApproxTokens) : "",
			value: working.toolResultMaxApproxTokens,
			assign: (value) => {
				working.toolResultMaxApproxTokens = value;
			},
		},
		{
			title: "Tool result char budget (blank = off)",
			placeholder: working.toolResultMaxChars ? String(working.toolResultMaxChars) : "",
			value: working.toolResultMaxChars,
			assign: (value) => {
				working.toolResultMaxChars = value;
			},
		},
		{
			title: "Tool result line budget (blank = off)",
			placeholder: working.toolResultMaxLines ? String(working.toolResultMaxLines) : "",
			value: working.toolResultMaxLines,
			assign: (value) => {
				working.toolResultMaxLines = value;
			},
		},
	];

	for (const prompt of prompts) {
		const value = await ctx.ui.input(prompt.title, prompt.placeholder);
		if (value === undefined) return undefined;
		prompt.assign(parsePositiveInt(value));
	}

	return sanitizeConfig(working);
}

async function configureAdvanced(
	ctx: ExtensionCommandContext,
	baseConfig: DistillConfig,
	sourceCounts: Record<DistillCategory, number>,
): Promise<DistillConfig | undefined> {
	const categories = await configureCategories(ctx, baseConfig, sourceCounts);
	if (!categories) return undefined;
	const filtered = await configureToolFilter(ctx, categories);
	if (!filtered) return undefined;
	return configureBudgets(ctx, filtered);
}

async function confirmDistill(
	ctx: ExtensionCommandContext,
	presetLabel: string,
	config: DistillConfig,
	stats: DistillStats,
): Promise<boolean> {
	const lines = [...configSummary(config), "", ...statsSummary(stats)].join("\n");
	return ctx.ui.confirm(`Create reduced session (${presetLabel})?`, lines);
}

function sessionNameForDistill(ctx: ExtensionCommandContext, label: string): string {
	const currentName = ctx.sessionManager.getSessionName();
	return currentName ? `${currentName} [reduce:${label}]` : `reduce:${label}`;
}

async function createDistilledSession(
	ctx: ExtensionCommandContext,
	label: string,
	config: DistillConfig,
	distilledMessages: Array<Message | LocalCustomMessage | BashExecutionMessage>,
	stats: DistillStats,
) {
	const sourceSessionFile = ctx.sessionManager.getSessionFile();
	const sourceLeafId = ctx.sessionManager.getLeafId();
	const sourceContext = buildSessionContext(ctx.sessionManager.getEntries(), sourceLeafId);
	const sessionName = sessionNameForDistill(ctx, label);

	await ctx.waitForIdle();
	const result = await ctx.newSession({
		parentSession: sourceSessionFile,
		setup: async (sessionManager) => {
			sessionManager.appendCustomEntry("reduce-source", {
				sourceSessionFile,
				sourceLeafId,
				label,
				config,
				stats,
				createdAt: new Date().toISOString(),
			});

			if (sourceContext.model) {
				sessionManager.appendModelChange(sourceContext.model.provider, sourceContext.model.modelId);
			}
			sessionManager.appendThinkingLevelChange(sourceContext.thinkingLevel);

			for (const message of distilledMessages) {
				sessionManager.appendMessage(message);
			}

			sessionManager.appendSessionInfo(sessionName);
		},
	});

	if (result.cancelled) {
		ctx.ui.notify("Reduce cancelled", "info");
		return;
	}

	ctx.ui.notify(compactFinalSummary(sessionName, stats), "info");
}

export default function distillExtension(pi: ExtensionAPI) {
	let state: DistillState = { lastConfig: cloneConfig(DEFAULT_CONFIG) };

	const syncState = (ctx: ExtensionCommandContext) => {
		state = restoreState(ctx);
	};

	const runPreset = async (preset: DistillPreset, ctx: ExtensionCommandContext) => {
		const config = cloneConfig(PRESETS[preset].config);
		const sourceContext = buildSessionContext(ctx.sessionManager.getEntries(), ctx.sessionManager.getLeafId());
		const distilled = distillMessages(sourceContext.messages, config);
		if (distilled.messages.length === 0) {
			ctx.ui.notify("That preset removed everything; try /reduce-advanced", "warning");
			return;
		}
		const confirmed = await confirmDistill(ctx, preset, config, distilled.stats);
		if (!confirmed) return;
		state = { lastConfig: cloneConfig(config), lastPreset: preset };
		pi.appendEntry(STATE_ENTRY_TYPE, state);
		await createDistilledSession(ctx, preset, config, distilled.messages, distilled.stats);
	};

	const runAdvanced = async (ctx: ExtensionCommandContext) => {
		const sourceContext = buildSessionContext(ctx.sessionManager.getEntries(), ctx.sessionManager.getLeafId());
		const sourceCounts = computeSourceCategoryCounts(sourceContext.messages);
		const configured = await configureAdvanced(ctx, state.lastConfig, sourceCounts);
		if (!configured) {
			ctx.ui.notify("Reduce advanced cancelled", "info");
			return;
		}
		const distilled = distillMessages(sourceContext.messages, configured);
		if (distilled.messages.length === 0) {
			ctx.ui.notify("Current configuration removed everything", "warning");
			return;
		}
		const confirmed = await confirmDistill(ctx, "custom", configured, distilled.stats);
		if (!confirmed) return;
		state = { lastConfig: cloneConfig(configured), lastPreset: undefined };
		pi.appendEntry(STATE_ENTRY_TYPE, state);
		await createDistilledSession(ctx, "custom", configured, distilled.messages, distilled.stats);
	};

	const reduceCommand = {
		description: "Create a reduced fork of the current branch (usage: /reduce [chat|reasoning|tools|no-tools|advanced|last])",
		getArgumentCompletions: (prefix: string) => {
			const options = ["chat", "reasoning", "tools", "no-tools", "advanced", "last"];
			const filtered = options.filter((option) => option.startsWith(prefix.toLowerCase()));
			return filtered.length > 0 ? filtered.map((value) => ({ value, label: value })) : null;
		},
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			if (!ctx.hasUI) {
				ctx.ui.notify("/reduce requires interactive mode", "error");
				return;
			}
			syncState(ctx);
			const parsed = parsePreset(args);
			if (args.trim() && !parsed) {
				ctx.ui.notify("Usage: /reduce [chat|reasoning|tools|no-tools|advanced|last]", "error");
				return;
			}

			if (parsed === "last") {
				const distilled = distillMessages(
					buildSessionContext(ctx.sessionManager.getEntries(), ctx.sessionManager.getLeafId()).messages,
					state.lastConfig,
				);
				if (distilled.messages.length === 0) {
					ctx.ui.notify("Last config removes everything; use /reduce-advanced", "warning");
					return;
				}
				const label = state.lastPreset ?? "last";
				const confirmed = await confirmDistill(ctx, label, state.lastConfig, distilled.stats);
				if (!confirmed) return;
				pi.appendEntry(STATE_ENTRY_TYPE, state);
				await createDistilledSession(ctx, label, state.lastConfig, distilled.messages, distilled.stats);
				return;
			}

			if (parsed === "advanced") {
				await runAdvanced(ctx);
				return;
			}

			if (parsed) {
				await runPreset(parsed, ctx);
				return;
			}

			const chosen = await pickPreset(ctx);
			if (!chosen) return;
			if (chosen === "advanced") {
				await runAdvanced(ctx);
				return;
			}
			await runPreset(chosen, ctx);
		},
	};

	const reduceAdvancedCommand = {
		description: "Open the full reduce wizard with message-type and tool filters",
		handler: async (_args: string, ctx: ExtensionCommandContext) => {
			if (!ctx.hasUI) {
				ctx.ui.notify("/reduce-advanced requires interactive mode", "error");
				return;
			}
			syncState(ctx);
			await runAdvanced(ctx);
		},
	};

	pi.registerCommand("reduce", reduceCommand);
	pi.registerCommand("reduce-advanced", reduceAdvancedCommand);

	pi.on("session_start", async (_event, ctx) => {
		state = restoreState(ctx as ExtensionCommandContext);
	});
	pi.on("session_switch", async (_event, ctx) => {
		state = restoreState(ctx as ExtensionCommandContext);
	});
	pi.on("session_tree", async (_event, ctx) => {
		state = restoreState(ctx as ExtensionCommandContext);
	});
	pi.on("session_fork", async (_event, ctx) => {
		state = restoreState(ctx as ExtensionCommandContext);
	});
}
