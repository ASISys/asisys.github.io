---
layout: about
title: Home
permalink: /
subtitle: >
  <b>Github Repo:</b> <a href="https://github.com/ASISys">github.com/ASISys</a>

selected_papers: false # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new post items
  limit: 3 # leave blank to include all the blog posts
---

Welcome to ASISys – an open-source organization dedicated to advancing system research and development in Artificial Super Intelligence (ASI). While ASI has not yet been fully realized, our vision is to create foundational systems and techniques that push the boundaries of current AI and lay the groundwork for the future emergence of ASI.

We focus on scalable, efficient, and adaptive AI systems that evolve over time, improving the efficacy and efficiency of both AI training and serving. Our work includes developing architectures, systems, algorithms, and tools that are essential for the transition from narrow AI to super intelligent systems.

---

## Projects

{% assign sorted_projects = site.projects | sort: "importance" %}

  <!-- Generate cards for each project -->

<div class="projects">
{% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
{% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
{% endif %}
</div>

---

## Publications

<div class="publications">

{% bibliography --group_by none %}

</div>

---
