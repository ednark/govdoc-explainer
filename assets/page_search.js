function filterList(query, results) {

    let matchingTitles = results.map(result => result.ref);
    let listItems = document.querySelectorAll('#nav-menu-standards li');

    for (let i = 0; i < listItems.length; i++) {
        let li = listItems[i];
        let a = li.querySelector('a');

        if ( !a || !a.textContent || a.textContent.toLowerCase().includes(query.toLowerCase())) {
            li.style.display = 'block';
            continue;
        }

        matchFound = false;
        for (let j = 0; j < matchingTitles.length; j++) {
            if (a.textContent.toLowerCase() == matchingTitles[j].toLowerCase()) {
                matchFound = true;
                break;
            }
        }
        li.style.display = matchFound ? 'block' : 'none';
    }
}


document.addEventListener("DOMContentLoaded", async function() {
    let lunrIndex = null;
    try {
        let indexJson = await fetch('../../assets/lunr_index.json');
        if ( indexJson.status !== 200 ) {
            throw new Error(`Can't load lunr index: ${indexJson.status} ${indexJson.statusText}`);
        }
        let indexData = await indexJson.json();
        lunrIndex = lunr.Index.load(indexData);
    } catch (error) {
        console.log(error)
    }
    if ( lunrIndex ) {
        window.lunrIndex = lunrIndex

        let search = document.createElement('div')
        search.classList.add('search-wrapper')
        search.id = 'nav-menu-search'
        let input = document.createElement('input')
        input.classList.add('search')
        input.setAttribute('placeholder', '... search for standard')
        input.addEventListener('input', function() {
            let lunrQuery = this.value
            if ( lunrQuery ) {
                let lunrResults = lunrIndex.search(lunrQuery)
                filterList( lunrQuery, lunrResults )
            }
        })
        search.appendChild(input)
        
        let standardsList = document.querySelector('#nav-menu-standards')
        let standardsContainer = standardsList.parentElement
        standardsContainer.insertBefore(search, standardsList)

    }

})  