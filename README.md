# 南陔的个人主页

基于 [Hydejack](https://hydejack.com/) v9 免费版 + Jekyll 搭建，托管在 GitHub Pages。
文章列表用 xinfang.site 风格的「emoji 折叠目录」组织。

线上：<https://Lukad77.github.io>

---

## 快速上手：**只想写一篇新文章**

1. 到 `_posts/` 目录下新建文件，命名 **必须**是 `YYYY-MM-DD-slug.md`
2. 文件开头写 front matter，最重要的是 `category` 字段：

   ```yaml
   ---
   title: "文章标题"
   date: 2026-07-08 10:00:00 +0800
   category: recsys           # ← 分类 key，见下表
   tags: [召回, DAG]           # ← 可选，标签
   description: >-
     一句话摘要，会显示在列表和 SEO。
   ---
   ```

3. front matter 下面正常写 Markdown
4. `git add . && git commit -m "post: xxx" && git push`
5. 等 GitHub Pages 构建完（大约 1~2 分钟），访问站点就能看到新文章自动归入对应 emoji 文件夹

**分类 key 表**（对应 `_data/categories.yml`）：

| key         | Emoji | 名称                    |
| ----------- | ----- | ---------------------- |
| `recsys`    | 🤖    | 推荐系统 / 机器学习      |
| `cpp-inner` | ⚙️    | C++ 内功                |
| `embedded`  | 🔌    | 嵌入式 / EE             |
| `thinking`  | 🧠    | 系统思考 / 方法论        |
| `life`      | 📖    | 生活随笔                |

> 想改 emoji / 分类名 / 顺序，直接编辑 `_data/categories.yml`，无需改代码。
> 想加新分类：在 `_data/categories.yml` 加一条，就能被 include 自动识别。

---

## 目录结构

```
Lukad77.github.io/
├── _config.yml                  # 全局配置（主题、作者、URL 等）
├── Gemfile                      # Ruby 依赖
├── index.md                     # 首页（折叠目录 + 最近更新）
├── about.md                     # 关于我
├── archive.md                   # 按年份归档
│
├── _posts/                      # 所有文章都放这里
│   ├── 2026-07-07-recall-vs-rank.md
│   ├── 2026-07-06-cpp-rvo.md
│   └── ...
│
├── _data/
│   └── categories.yml           # 分类定义（emoji + 名称 + key）
│
├── _includes/
│   └── categories-tree.html     # 折叠目录组件（首页调用）
│
├── _sass/
│   └── my-inline.scss           # 自定义样式（折叠目录 + 粉紫青渐变）
│
├── assets/
│   └── img/avatar.png           # 头像
│
└── _backup/
    └── index-v1.html            # v1 手作单页首页（保留备份）
```

---

## 本地预览

第一次：

```bash
cd Lukad77.github.io
bundle install
bundle exec jekyll serve
```

访问 <http://localhost:4000> 即可看到效果。

之后只要：

```bash
bundle exec jekyll serve
```

修改 `.md` 会自动热重载，改 `_config.yml` 需要重启。

---

## 常用操作速查

| 想做的事             | 怎么做                                                    |
| ------------------- | -------------------------------------------------------- |
| 写一篇新文章         | 在 `_posts/` 新建 `YYYY-MM-DD-xxx.md`，front matter 写 `category` |
| 增加/修改分类        | 编辑 `_data/categories.yml`                              |
| 修改折叠目录样式     | 编辑 `_sass/my-inline.scss` 的 `.cat-tree` 部分          |
| 修改站点标题/描述    | 编辑 `_config.yml` 顶部                                  |
| 修改"关于"页         | 编辑 `about.md`                                          |
| 一篇文章标"更新时间" | 在 front matter 加 `last_modified_at: 2026-07-08`         |
| 一篇文章暂时不发布   | front matter 加 `published: false`，或者放到 `_drafts/`   |

---

## 主题相关

- Hydejack 官方文档：<https://hydejack.com/docs/>
- 用的是 `remote_theme: hydecorp/hydejack@v9.2.1`（免费版）
- 想升级：改 `_config.yml` 里的版本号即可
- 想覆盖某个主题 layout：把它复制到本地 `_layouts/` 下同名文件即可
