document.addEventListener("DOMContentLoaded", function() {
    function toggleAccordion( button ) {
        button.classList.toggle('active');
        const expanded = button.classList.contains('active');
        button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        const content = button.nextElementSibling;
        if (content.style.maxHeight) {
            content.style.maxHeight = null;
        } else {
            content.style.maxHeight = content.scrollHeight + "px";
        }
    }
    document.querySelectorAll('.accordion-header').forEach(button => {
        button.addEventListener('click', () => { toggleAccordion(button) });
    });

    document.querySelectorAll('button').forEach(button => {
        if (button.textContent.trim() === 'Punchline Summary') {
            toggleAccordion(button)
        }
    });

});
