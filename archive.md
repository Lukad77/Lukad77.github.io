---
layout: page
title: 归档
menu: true
order: 3
description: >-
  按年份倒序的全部文章列表
---

{% assign posts_by_year = site.posts | group_by_exp:"p","p.date | date: '%Y'" %}

{% for year in posts_by_year %}
## {{ year.name }}

<ul class="post-list">
{% for post in year.items %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <span class="post-date"> · {{ post.date | date: "%m-%d" }}</span>
  </li>
{% endfor %}
</ul>

{% endfor %}
