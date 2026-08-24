import { defineCollection, z } from "astro:content"
import { glob } from "astro/loaders"

const posts = defineCollection({
  loader: glob({ base: "./src/content/posts", pattern: "**/*.md" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    tags: z.array(z.string()).default([]),
  }),
})

export const collections = { posts }
