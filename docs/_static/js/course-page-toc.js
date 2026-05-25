(function () {
  "use strict";

  function headingText(heading) {
    const clone = heading.cloneNode(true);
    clone.querySelectorAll(".headerlink").forEach((link) => link.remove());
    return clone.textContent.trim();
  }

  function sectionIdForHeading(heading) {
    const section = heading.closest("section[id]");
    return section ? section.id : "";
  }

  function createItem(heading) {
    const sectionId = sectionIdForHeading(heading);
    const text = headingText(heading);

    if (!sectionId || !text) {
      return null;
    }

    const item = document.createElement("li");
    const link = document.createElement("a");

    item.className = `course-page-toc__item course-page-toc__item--${heading.tagName.toLowerCase()}`;
    link.href = `#${sectionId}`;
    link.textContent = text;
    item.append(link);

    return item;
  }

  function initPageToc() {
    const documentWrapper = document.querySelector(".documentwrapper");
    const body = document.querySelector(".body[role='main']");

    if (!documentWrapper || !body) {
      return;
    }

    const items = [...body.querySelectorAll("section[id] > h2, section[id] > h3")]
      .map(createItem)
      .filter(Boolean);

    if (items.length < 2) {
      return;
    }

    const toc = document.createElement("nav");
    const title = document.createElement("div");
    const list = document.createElement("ol");

    toc.className = "course-page-toc";
    toc.setAttribute("aria-label", "Page table of contents");
    title.className = "course-page-toc__title";
    title.textContent = "On this page";
    list.className = "course-page-toc__list";
    list.append(...items);
    toc.append(title, list);

    documentWrapper.prepend(toc);
  }

  document.addEventListener("DOMContentLoaded", initPageToc);
})();
