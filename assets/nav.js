document.addEventListener("DOMContentLoaded", function() {

    if ( !sources ) {
        sources = {"standard_name":"./sources/standard/index.html"};
    }

    // Create the nav menu
    const menu = document.getElementById('nav-menu');
    const toggle = document.getElementById('nav-menu-toggle');
    const standards = document.getElementById("nav-menu-standards")

    const currPageEl = document.querySelector("h1");
    const currPage = currPageEl ? currPageEl.textContent : "";

    // Loop through the sources object and create a list item for each source
    for (let key in sources) {
        if (sources.hasOwnProperty(key)) {
            let sourceName = key;
            let sourceLink = sources[key];

            let slistItem = document.createElement("li");
            let slink = document.createElement("a");

            if (sourceName == currPage) {
                slink = document.createElement("span")
                slistItem.classList.add("current");
            }

            slink.href = sourceLink;
            slink.textContent = sourceName;

            slistItem.appendChild(slink);
            standards.appendChild(slistItem);
        }
    }


    toggle.addEventListener('click', () => {
        menu.classList.toggle('active');
    });

    // Close accordion sections when clicking outside
    document.addEventListener('click', function(event) {
        if (!menu.contains(event.target) && event.target !== toggle && menu.classList.contains('active') ) {
            menu.classList.remove('active');
            toggle.classList.remove('active');
            const content = toggle.nextElementSibling;
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
            }
        }
    });

});
