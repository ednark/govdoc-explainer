document.addEventListener("DOMContentLoaded", function() {
    function toggleAccordion( button ) {
        button.classList.toggle('active');
        const expanded = button.classList.contains('active');
        button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        const content = button.nextElementSibling;
        if (content.style.maxHeight) {
            content.style.maxHeight = null;
        } else {
            // slack guard: scrollHeight rounds sub-pixel text down and can be measured
            // before the webfont swaps, which otherwise crops the first/last text lines
            content.style.maxHeight = (content.scrollHeight + 24) + "px";
        }
    }
    document.querySelectorAll('.accordion-header').forEach(button => {
        button.addEventListener('click', () => { toggleAccordion(button) });
    });

    document.querySelectorAll('button').forEach(button => {
        if (['Punchline Summary', 'Executive Brief'].includes(button.textContent.trim())) {
            toggleAccordion(button)
        }
    });

});
