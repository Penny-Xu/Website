import { defineCollection, z } from "astro:content"
import { glob } from "astro/loaders"

const posts = defineCollection({
  loader: glob({ base: "./src/content/posts", pattern: "**/*.md" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    tags: z.array(z.string()).default([]),
    // Optional: used for the meta description and social card. Falls back to
    // the opening of the post body when omitted.
    description: z.string().optional(),
  }),
})

export const collections = { posts }
