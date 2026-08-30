document.querySelectorAll("[data-tab-group]").forEach((group) => {
  const tabs = group.querySelectorAll(".benchmark-tab");
  const panels = group.querySelectorAll(".benchmark-panel");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const selectedPanel = tab.dataset.panel;

      tabs.forEach((item) => {
        const isSelected = item === tab;
        item.classList.toggle("active", isSelected);
        item.setAttribute("aria-selected", String(isSelected));
      });

      panels.forEach((panel) => {
        panel.hidden = panel.id !== selectedPanel;
      });
    });
  });
});
