---
layout: page
title: 南陔的知识小屋
description: >-
  从模拟电路的一颗三极管，到推荐系统里的一次点击 —— 都是在追同一件事：让信号找到对的人。
sitemap: false
---

## 关于我

📕 **从模拟电路的一颗三极管，到推荐系统里的一次点击 —— 都是在追同一件事：让信号找到对的人。**

📘 An EE-born Recommender Engineer, wandering between hardware intuition and data-driven models.

📙 打球、玩 CS2、写嵌入式、也算算数 —— 相信理科生的浪漫在于把复杂的东西拆开看。

---

## 📚 知识小屋（按分类）

{% include categories-tree.html %}

---

## 📝 最近更新

<ul class="post-list">
{% for post in site.posts limit:5 %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <span class="post-date"> · {{ post.date | date: "%Y-%m-%d" }}</span>
  </li>
{% endfor %}
</ul>
