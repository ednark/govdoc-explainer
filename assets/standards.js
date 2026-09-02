document.addEventListener("DOMContentLoaded", function() {
    function toggleAccordion( button ) {
        button.classList.toggle('active');
        const expanded = button.classList.contains('active');
        button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        const content = button.nextElementSibling;
        content.classList.toggle('open', expanded);
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
