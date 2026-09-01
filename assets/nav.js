document.addEventListener("DOMContentLoaded", function() {

    const menu = document.getElementById('nav-menu');
    const toggle = document.getElementById('nav-menu-toggle');
    const standards = document.getElementById("nav-menu-standards");
    if (!menu || !toggle || !standards || typeof sources === 'undefined') return;

    // tolerate both nested {category: {doc: path}} and legacy flat {doc: path} formats
    if (!Object.values(sources).some(v => typeof v === 'object')) {
        sources = { 'Standards': sources };
    }

    const currPageEl = document.querySelector("h1");
    const currPage = currPageEl ? currPageEl.textContent : "";

    const categories = Object.keys(sources).sort();
    for (const category of categories) {
        const docs = sources[category];
        const catLi = document.createElement('li');
        catLi.className = 'nav-category';

        const catBtn = document.createElement('button');
        catBtn.className = 'nav-category-toggle';
        catBtn.type = 'button';
        catBtn.setAttribute('aria-expanded', 'false');
        catBtn.textContent = category;

        const docUl = document.createElement('ul');
        docUl.className = 'nav-category-docs';

        let containsCurrent = false;
        for (const docName of Object.keys(docs).sort()) {
            const li = document.createElement('li');
            const link = document.createElement(docName === currPage ? 'span' : 'a');
            if (docName === currPage) {
                li.classList.add('current');
                containsCurrent = true;
            }
            link.href = docs[docName];
            link.textContent = docName;
            li.appendChild(link);
            docUl.appendChild(li);
        }

        if (containsCurrent) {
            catLi.classList.add('open');
            catBtn.setAttribute('aria-expanded', 'true');
        }

        catBtn.addEventListener('click', () => {
            const open = catLi.classList.toggle('open');
            catBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
        });

        catLi.appendChild(catBtn);
        catLi.appendChild(docUl);
        standards.appendChild(catLi);
    }

    toggle.addEventListener('click', () => {
        menu.classList.toggle('active');
    });

    document.addEventListener('click', function(event) {
        if (!menu.contains(event.target) && menu.classList.contains('active')) {
            menu.classList.remove('active');
            toggle.classList.remove('active');
        }
    });

});
