import DOMPurify from "dompurify";
import { marked } from "marked";

const MAX_MARKDOWN_CACHE_ENTRIES = 500;
const markdownCache = new Map<string, string>();

export function renderMarkdown(text: string) {
  if (!text) {
    return "";
  }
  const cached = markdownCache.get(text);
  if (cached != null) {
    markdownCache.delete(text);
    markdownCache.set(text, cached);
    return cached;
  }
  const raw = marked.parse(text, { async: false, breaks: true }) as string;
  const sanitized = DOMPurify.sanitize(raw);
  markdownCache.set(text, sanitized);
  if (markdownCache.size > MAX_MARKDOWN_CACHE_ENTRIES) {
    const oldestKey = markdownCache.keys().next().value;
    if (oldestKey) {
      markdownCache.delete(oldestKey);
    }
  }
  return sanitized;
}
