import type { RenderableTreeNodes } from "@markdoc/markdoc";

import markdoc from "@markdoc/markdoc";
import * as React from "react";
import { Fence } from "./fence";
import { Callout } from "./callout";
import { BulletListItem } from "./bullet-list-item";

type Props = {
  content: RenderableTreeNodes;
};

const { renderers } = markdoc;

export function MarkdownView({ content }: Readonly<Props>) {
  return (
    <>
      {renderers.react(content, React, {
        components: { Callout, Fence, BulletListItem },
      })}
    </>
  );
}
