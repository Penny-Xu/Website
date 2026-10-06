// Build a plain-text meta description from raw markdown. Used as a fallback
// when a post has no explicit `description` in its frontmatter.
export function excerpt(markdown, limit = 160) {
  const text = markdown
    .replace(/^---[\s\S]*?---/, "") // frontmatter, if present
    .replace(/```[\s\S]*?```/g, "") // fenced code
    .replace(/^>.*$/gm, "") // blockquotes
    .replace(/^\s*\|.*$/gm, "") // tables
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "") // images
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1") // links -> text
    .replace(/<[^>]+>/g, "") // raw html
    .replace(/[#*_`$]/g, "")
    .replace(/\s+/g, " ")
    .trim()

  if (text.length <= limit) return text

  const cut = text.slice(0, limit)
  const lastSpace = cut.lastIndexOf(" ")
  return `${cut.slice(0, lastSpace > 0 ? lastSpace : limit).trimEnd()}…`
}
