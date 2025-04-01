import Markdoc, { type RenderableTreeNodes } from "@markdoc/markdoc";

const fence = {
  render: "Fence",
  attributes: {
    language: {
      type: String,
    },
  },
};

const blockquote = {
  render: "Callout",
  attributes: {
    citation: {
      type: String,
      required: false,
    },
  },
  children: [],
};

const item = {
  render: "BulletListItem",
  attributes: {
    ordered: { type: Boolean, default: false },
  },
  children: [],
};

export function markdownParser(markdown: string): RenderableTreeNodes {
  return Markdoc.transform(Markdoc.parse(markdown), {
    nodes: { fence, blockquote, item },
  });
}
