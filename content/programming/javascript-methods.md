---
tags: [programming]
title: Javascripts Methods
publishedAt: "2025-02-20"
summary: This is a summary about some methods in javascript
---

## `.sort()`

With this method we can sort an array. This method sorts the elements `in place`, which means that the method does not create a new array, but operates directly on the input data and returns a reference to the same array.

For example:

```js
const entries = [
  {
    title: "movie 1",
    publishedAt: "2023-01-23",
  },
  {
    title: "movie 2",
    publishedAt: "2023-01-21",
  },
  {
    title: "movie 3",
    publishedAt: "2023-01-22",
  },
];

const sortedBlogs = allBlogs.sort(
  (a, b) =>
    new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime()
);
```
