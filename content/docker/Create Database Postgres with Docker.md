---
title: "Create Database Postgres with Docker"
publishedAt: "2025-02-21"
summary: "In the ever-evolving landscape of software development, the debate between dynamic and static typing continues to be a hot topic."
tags:
  - docker
  - programming
---

This command get de last version of postgres

```
docker pull postrgres
```

Then we run de following command

```
docker run --name example-db -d -p 5432:5432 -e POSTGRES_PASSWORD=sarasa postgres
```
