
model = null;
embeddingsData = [];

async function loadUSEModel() {
    // Load the Universal Sentence Encoder model
    const model = await use.load();
    return model;
}

async function loadEmbeddings() {
    // Load the embeddings from the JSON file
    const response = await fetch("./assets/embedding.json"); // Adjust the path as necessary
    const data = await response.json();
    return data;
}

async function generateEmbedding(model, text) {
    // Generate embeddings for the given text
    const embeddings = await model.embed([text]);
    return embeddings.squeeze();
}

function cosineSimilarityUse(a, b) {
    return tf.tidy(() => {
        // Ensure both tensors are 2D and have the same shape
        const a2d = a.reshape([1, -1]);
        const b2d = b.reshape([1, -1]);

        // Normalize the vectors
        const a_normalized = a2d.div(a2d.norm());
        const b_normalized = b2d.div(b2d.norm());

        // Compute dot product
        const dot_product = a_normalized.matMul(b_normalized.transpose());

        // The result is a 1x1 tensor, so we need to get its value
        return dot_product.squeeze();
    });
}

function matchSimilarity(setA, setB) {
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    return intersection.size / Math.min( setA.size, setB.size );
}

async function hybridSimilarity(id, queryEmbedding, entryEmbedding, queryText, entryText) {
    if ( !queryEmbedding || !entryEmbedding || !entryText || !queryText ) {
        return 0;
    }
    if ( entryEmbedding.length == 512 ) {
        entryEmbedding = [entryEmbedding]
    }
    let itemEmbedding = tf.tensor2d(entryEmbedding)
    let similarity = await cosineSimilarityUse( queryEmbedding, itemEmbedding )
    let similarityData = await similarity.data()
    let cosineSim = similarityData[0] || 0
    
    // word matching
    let queryWords = queryText.split(/\s+/).filter( w => w.length && !stopwords_en.includes(w.toLowerCase()) );
    let entryWords = entryText.split(/\s+/).filter( w => w.length && !stopwords_en.includes(w.toLowerCase()) );
    queryWords = new Set( queryWords.map( x => metaphone(stemmer(x.toLowerCase()) )) );
    entryWords = new Set( entryWords.map( x => metaphone(stemmer(x.toLowerCase()) )) );
    const matchSim = matchSimilarity(queryWords, entryWords);
    
    return cosineSim + matchSim
}

async function embeddings_search(query) {
    similarities = [];
    const queryEmbedding = await generateEmbedding(model, query);
    try {
        similarities = await Promise.all(embeddingsData.map(async item => { return {
            id: item.id,
            title: item.title,
            text: item.body,
            similarity: await hybridSimilarity(item.id, queryEmbedding, item.embedding, query, item.body)
        }}));
    } catch (error) {
        console.log(error)
    }
    // console.log(similarities)
    similarities.sort((a, b) => b.similarity - a.similarity);
    similarities = similarities.filter(x => x.similarity >= 0.4);

    return similarities
}


function filterList(query, results) {

    let matchingTitles = results.map(result => result.title);
    let listItems = document.querySelectorAll('#nav-menu-standards li');

    for (let i = 0; i < listItems.length; i++) {
        let li = listItems[i];
        let a = li.querySelector('a');

        if (a && a.textContent) {
            if (a.textContent.toLowerCase().includes(query.toLowerCase())) {
                li.style.display = 'block';
                continue;
            }
            let matchFound = false;
            for (let j = 0; j < matchingTitles.length; j++) {
                if (a.textContent == matchingTitles[j]) {
                    matchFound = true;
                    break;
                }
            }
            li.style.display = matchFound ? 'block' : 'none';
        }

    }
}

document.addEventListener("DOMContentLoaded", async function() {
    model = await loadUSEModel();
    embeddingsData = await loadEmbeddings();

    async function performSearch() {
        let query = this.value
        if ( query ) {
            let results = await embeddings_search(query)
            filterList( query, results )
        }
    }
    const debouncedSearch = debounce(performSearch, 300); // 300ms delay

    // Add the event listener to the input
    let input = document.createElement('input')
    input.addEventListener('input', debouncedSearch);
    

    if ( embeddingsData ) {
        let search = document.createElement('div')
        search.classList.add('search-wrapper')
        search.id = 'nav-menu-search'
        input.classList.add('search')
        input.setAttribute('placeholder', '... search')
        input.addEventListener('input', debouncedSearch)
        search.appendChild(input)
        
        // inject search box at top of nav
        let standardsList = document.querySelector('#nav-menu-standards')
        let standardsContainer = standardsList.parentElement
        standardsContainer.insertBefore(search, standardsList)
    }

    function debounce(func, delay) {
        let timeoutId;
        return function (...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    }

})

var stopwords_en = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"];
var stemmer=function(){function h(){}function i(){console.log(Array.prototype.slice.call(arguments).join(" "))}var j={ational:"ate",tional:"tion",enci:"ence",anci:"ance",izer:"ize",bli:"ble",alli:"al",entli:"ent",eli:"e",ousli:"ous",ization:"ize",ation:"ate",ator:"ate",alism:"al",iveness:"ive",fulness:"ful",ousness:"ous",aliti:"al",iviti:"ive",biliti:"ble",logi:"log"},k={icate:"ic",ative:"",alize:"al",iciti:"ic",ical:"ic",ful:"",ness:""};return function(a,l){var d,b,g,c,f,e;e=l?i:h;if(3>a.length)return a;
    g=a.substr(0,1);"y"==g&&(a=g.toUpperCase()+a.substr(1));c=/^(.+?)(ss|i)es$/;b=/^(.+?)([^s])s$/;c.test(a)?(a=a.replace(c,"$1$2"),e("1a",c,a)):b.test(a)&&(a=a.replace(b,"$1$2"),e("1a",b,a));c=/^(.+?)eed$/;b=/^(.+?)(ed|ing)$/;c.test(a)?(b=c.exec(a),c=/^([^aeiou][^aeiouy]*)?[aeiouy][aeiou]*[^aeiou][^aeiouy]*/,c.test(b[1])&&(c=/.$/,a=a.replace(c,""),e("1b",c,a))):b.test(a)&&(b=b.exec(a),d=b[1],b=/^([^aeiou][^aeiouy]*)?[aeiouy]/,b.test(d)&&(a=d,e("1b",b,a),b=/(at|bl|iz)$/,f=/([^aeiouylsz])\1$/,d=/^[^aeiou][^aeiouy]*[aeiouy][^aeiouwxy]$/,
    b.test(a)?(a+="e",e("1b",b,a)):f.test(a)?(c=/.$/,a=a.replace(c,""),e("1b",f,a)):d.test(a)&&(a+="e",e("1b",d,a))));c=/^(.*[aeiouy].*)y$/;c.test(a)&&(b=c.exec(a),d=b[1],a=d+"i",e("1c",c,a));c=/^(.+?)(ational|tional|enci|anci|izer|bli|alli|entli|eli|ousli|ization|ation|ator|alism|iveness|fulness|ousness|aliti|iviti|biliti|logi)$/;c.test(a)&&(b=c.exec(a),d=b[1],b=b[2],c=/^([^aeiou][^aeiouy]*)?[aeiouy][aeiou]*[^aeiou][^aeiouy]*/,c.test(d)&&(a=d+j[b],e("2",c,a)));c=/^(.+?)(icate|ative|alize|iciti|ical|ful|ness)$/;
    c.test(a)&&(b=c.exec(a),d=b[1],b=b[2],c=/^([^aeiou][^aeiouy]*)?[aeiouy][aeiou]*[^aeiou][^aeiouy]*/,c.test(d)&&(a=d+k[b],e("3",c,a)));c=/^(.+?)(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ou|ism|ate|iti|ous|ive|ize)$/;b=/^(.+?)(s|t)(ion)$/;c.test(a)?(b=c.exec(a),d=b[1],c=/^([^aeiou][^aeiouy]*)?[aeiouy][aeiou]*[^aeiou][^aeiouy]*[aeiouy][aeiou]*[^aeiou][^aeiouy]*/,c.test(d)&&(a=d,e("4",c,a))):b.test(a)&&(b=b.exec(a),d=b[1]+b[2],b=/^([^aeiou][^aeiouy]*)?[aeiouy][aeiou]*[^aeiou][^aeiouy]*[aeiouy][aeiou]*[^aeiou][^aeiouy]*/,
    b.test(d)&&(a=d,e("4",b,a)));c=/^(.+?)e$/;if(c.test(a)&&(b=c.exec(a),d=b[1],c=/^([^aeiou][^aeiouy]*)?[aeiouy][aeiou]*[^aeiou][^aeiouy]*[aeiouy][aeiou]*[^aeiou][^aeiouy]*/,b=/^([^aeiou][^aeiouy]*)?[aeiouy][aeiou]*[^aeiou][^aeiouy]*([aeiouy][aeiou]*)?$/,f=/^[^aeiou][^aeiouy]*[aeiouy][^aeiouwxy]$/,c.test(d)||b.test(d)&&!f.test(d)))a=d,e("5",c,b,f,a);c=/ll$/;b=/^([^aeiou][^aeiouy]*)?[aeiouy][aeiou]*[^aeiou][^aeiouy]*[aeiouy][aeiou]*[^aeiou][^aeiouy]*/;c.test(a)&&b.test(a)&&(c=/.$/,a=a.replace(c,""),e("5",
    c,b,a));"y"==g&&(a=g.toLowerCase()+a.substr(1));return a}}();    
var metaphone=function( string )
{
    function string_at(string, start, length, list) 
    {
        if ((start <0) || (start >= string.length))
            return 0;
    
        for (var i=0, len=list.length; i<len; i++) {
            if (list[i] == string.substr(start, length))
            return 1;
        }
        return 0;
    }
    function is_vowel(string, pos)
    {
        return /[AEIOUY]/.test(string.substr(pos, 1));
    }
    function Slavo_Germanic(string) 
    {
        return /W|K|CZ|WITZ/.test(string);     
    }

    primary   = "";
    current   =  0;
    
    current  = 0;
    length   = string.length;
    last     = length - 1;
    original = string + "     ";

    original = original.toUpperCase();

    // skip this at beginning of word
    
    if (string_at(original, 0, 2, 
                        ['GN', 'KN', 'PN', 'WR', 'PS']))
        current++;

    // Initial 'X' is pronounced 'Z' e.g. 'Xavier'
    
    if (original.substr(0, 1) == 'X') {
        primary   += "S";   // 'Z' maps to 'S'
        current++;
    }

    // main loop

    while (primary.length < 4) {
        if (current >= length)
        break;

        switch (original.substr(current, 1)) {
        case 'A':
        case 'E':
        case 'I':
        case 'O':
        case 'U':
        case 'Y':
            if (current == 0) {
            // all init vowels now map to 'A'
            primary   += 'A';
            }
            current += 1;
            break;

        case 'B':
            // '-mb', e.g. "dumb", already skipped over ...
            primary   += 'P';

            if (original.substr(current + 1, 1) == 'B')
            current += 2;
            else
            current += 1;
            break;

        case 'Ç':
            primary   += 'S';
            current += 1;
            break;

        case 'C':
            // various gremanic
            if ((current > 1) 
                && !is_vowel(original, current - 2)
                && string_at(original, current - 1, 3, 
                        ["ACH"])
                && ((original.substr(current + 2, 1) != 'I')
                    && ((original.substr(current + 2, 1) != 'E')
                        || string_at(original, current - 2, 6, 
                                ["BACHER", "MACHER"])))) {

            primary   += 'K';
            current += 2;
            break;
            }

            // special case 'caesar'
            if ((current == 0) 
                && string_at(original, current, 6, 
                            ["CAESAR"])) {
            primary   += 'S';
            current += 2;
            break;
            }

            // italian 'chianti'
            if (string_at(original, current, 4, 
                            ["CHIA"])) {
            primary   += 'K';
            current += 2;
            break;
            }

            if (string_at(original, current, 2, 
                            ["CH"])) {

            // find 'michael'
            if ((current > 0)
                && string_at(original, current, 4, 
                            ["CHAE"])) {
                primary   += 'K';
                current += 2;
                break;
            }

            // greek roots e.g. 'chemistry', 'chorus'
            if ((current == 0)
                && (string_at(original, current + 1, 5, 
                            ["HARAC", "HARIS"])
                    || string_at(original, current + 1, 3, 
                                ["HOR", "HYM", "HIA", "HEM"]))
                && !string_at(original, 0, 5, ["CHORE"])) {
                primary   += 'K';
                current += 2;
                break;
            }

            // germanic, greek, or otherwise 'ch' for 'kh' sound
            if ((string_at(original, 0, 4, ["VAN ", "VON "])
                    || string_at(original, 0, 3, ["SCH"]))
                // 'architect' but not 'arch', orchestra', 'orchid'
                || string_at(original, current - 2, 6, 
                            ["ORCHES", "ARCHIT", "ORCHID"])
                || string_at(original, current + 2, 1, 
                            ["T", "S"])
                || ((string_at(original, current - 1, 1, 
                            ["A","O","U","E"])
                        || (current == 0))
                    // e.g. 'wachtler', 'weschsler', but not 'tichner'
                    && string_at(original, current + 2, 1, 
                            ["L","R","N","M","B","H","F","V","W"," "]))) {
                primary   += 'K';
            } else {
                if (current > 0) {
                if (string_at(original, 0, 2, ["MC"])) {
                    // e.g. 'McHugh'
                    primary   += 'K';
                } else {
                    primary   += 'X';
                }
                } else {
                primary   += 'X';
                }
            }
            current += 2;
            break;
            }

            // e.g. 'czerny'
            if (string_at(original, current, 2, ["CZ"])
                && !string_at(original, current -2, 4, 
                            ["WICZ"])) {
            primary   += 'S';
            current += 2;
            break;
            }

            // e.g. 'focaccia'
            if (string_at(original, current + 1, 3, 
                        ["CIA"])) {
            primary   += 'X';
            current += 3;
            break;
            }

            // double 'C', but not McClellan'
            if (string_at(original, current, 2, ["CC"])
                && !((current == 1) 
                    && (original.substr(0, 1) == 'M'))) {
            // 'bellocchio' but not 'bacchus'
            if (string_at(original, current + 2, 1,
                        ["I","E","H"])
                && !string_at(original, current + 2, 2,
                            ["HU"])) {
                // 'accident', 'accede', 'succeed'
                if (((current == 1)
                    && (original.substr(current - 1, 1) == 'A'))
                    || string_at(original, current - 1, 5,
                            ["UCCEE", "UCCES"])) {
                primary   += "KS";
                // 'bacci', 'bertucci', other italian
                } else {
                primary   += "X";
                }
                current += 3;
                break;
            } else {
                // Pierce's rule
                primary   += "K";
                current += 2;
                break;
            }
            }

            if (string_at(original, current, 2,
                        ["CK","CG","CQ"])) {
            primary   += "K";
            current += 2;
            break;
            }

            if (string_at(original, current, 2,
                        ["CI","CE","CY"])) {
            // italian vs. english
            if (string_at(original, current, 3,
                        ["CIO","CIE","CIA"])) {
                primary   += "S";
            } else {
                primary   += "S";
            }
            current += 2;
            break;
            }

            // else
            primary   += "K";

            // name sent in 'mac caffrey', 'mac gregor'
            if (string_at(original, current + 1, 2,
                        [" C"," Q"," G"])) {
            current += 3;
            } else {
            if (string_at(original, current + 1, 1,
                        ["C","K","Q"])
                && !string_at(original, current + 1, 2,
                            ["CE","CI"])) {
                current += 2;
            } else {
                current += 1;
            }
            }
            break;

        case 'D':
            if (string_at(original, current, 2,
                        ["DG"])) {
            if (string_at(original, current + 2, 1,
                        ["I","E","Y"])) {
                // e.g. 'edge'
                primary   += "J";
                current += 3;
                break;
            } else {
                // e.g. 'edgar'
                primary   += "TK";
                current += 2;
                break;
            }
            }

            if (string_at(original, current, 2,
                        ["DT","DD"])) {
            primary   += "T";
            current += 2;
            break;
            }

            // else
            primary   += "T";
            current += 1;
            break;

        case 'F':
            if (original.substr(current + 1, 1) == 'F')
            current += 2;
            else
            current += 1;
            primary   += "F";
            break;

        case 'G':
            if (original.substr(current + 1, 1) == 'H') {
            if ((current > 0) 
                && !is_vowel(original, current - 1)) {
                primary   += "K";
                current += 2;
                break;
            }

            if (current < 3) {
                // 'ghislane', 'ghiradelli'
                if (current == 0) {
                if (original.substr(current + 2, 1) == 'I') {
                    primary   += "J";
                } else {
                    primary   += "K";
                }
                current += 2;
                break;
                }
            }

            // Parker's rule (with some further refinements) - e.g. 'hugh'
            if (((current > 1)
                    && string_at(original, current - 2, 1,
                            ["B","H","D"]))
                // e.g. 'bough'
                || ((current > 2)
                    &&  string_at(original, current - 3, 1,
                                ["B","H","D"]))
                // e.g. 'broughton'
                || ((current > 3)
                    && string_at(original, current - 4, 1,
                                ["B","H"]))) {
                current += 2;
                break;
            } else {
                // e.g. 'laugh', 'McLaughlin', 'cough', 'gough', 'rough', 'tough'
                if ((current > 2)
                    && (original.substr(current - 1, 1) == 'U')
                    && string_at(original, current - 3, 1,
                            ["C","G","L","R","T"])) {
                primary   += "F";
                } else if ( (current > 0) && original.substr(current - 1, 1) != 'I') {
                primary   += "K";
                }
                current += 2;
                break;
            }
            }

            if (original.substr(current + 1, 1) == 'N') {
            if ((current == 1) && is_vowel(original, 0)
                && !Slavo_Germanic(original)) {
                primary   += "KN";
            } else {
                // not e.g. 'cagney'
                if (!string_at(original, current + 2, 2,
                            ["EY"])
                    && (original.substr(current + 1) != "Y")
                    && !Slavo_Germanic(original)) {
                    primary   += "N";
                } else {
                    primary   += "KN";
                }
            }
            current += 2;
            break;
            }

            // 'tagliaro'
            if (string_at(original, current + 1, 2,
                        ["LI"])
                && !Slavo_Germanic(original)) {
            primary   += "KL";
            current += 2;
            break;
            }

            // -ges-, -gep-, -gel- at beginning
            if ((current == 0)
                && ((original.substr(current + 1, 1) == 'Y')
                    || string_at(original, current + 1, 2,
                            ["ES","EP","EB","EL","EY","IB","IL","IN","IE",
                                    "EI","ER"]))) {
            primary   += "K";
            current += 2;
            break;
            }

            // -ger-, -gy-
            if ((string_at(original, current + 1, 2,
                        ["ER"])
                || (original.substr(current + 1, 1) == 'Y'))
                && !string_at(original, 0, 6,
                            ["DANGER","RANGER","MANGER"])
                && !string_at(original, current -1, 1,
                            ["E", "I"])
                && !string_at(original, current -1, 3,
                            ["RGY","OGY"])) {
            primary   += "K";
            current += 2;
            break;
            }

            // italian e.g. 'biaggi'
            if (string_at(original, current + 1, 1,
                        ["E","I","Y"])
                || string_at(original, current -1, 4,
                        ["AGGI","OGGI"])) {
            // obvious germanic
            if ((string_at(original, 0, 4, ["VAN ", "VON "])
                    || string_at(original, 0, 3, ["SCH"]))
                || string_at(original, current + 1, 2,
                            ["ET"])) {
                primary   += "K";
            } else {
                primary   += "J";
            }
            current += 2;
            break;
            }

            if (original.substr(current +1, 1) == 'G')
            current += 2;
            else
            current += 1;
            primary   += 'K';
            break;

        case 'H':
            // only keep if first & before vowel or btw. 2 vowels
            if (((current == 0) || 
                is_vowel(original, current - 1))
                && is_vowel(original, current + 1)) {
            primary   += 'H';
            current += 2;
            } else
            current += 1;
            break;

        case 'J':
            // obvious spanish, 'jose', 'san jacinto'
            if (string_at(original, current, 4,
                        ["JOSE"])
                || string_at(original, 0, 4, ["SAN "])) {
            if (((current == 0)
                    && (original.substr(current + 4, 1) == ' '))
                || string_at(original, 0, 4, ["SAN "])) {
                primary   += 'H';
            } else {
                primary   += "J";
            }
            current += 1;
            break;
            }

            if ((current == 0)
                && !string_at(original, current, 4,
                        ["JOSE"])) {
            primary   += 'J';  // Yankelovich/Jankelowicz
            } else {
            // spanish pron. of .e.g. 'bajador'
            if (is_vowel(original, current - 1)
                && !Slavo_Germanic(original)
                && ((original.substr(current + 1, 1) == 'A')
                    || (original.substr(current + 1, 1) == 'O'))) {
                primary   += "J";
            } else {
                if (current == last) {
                primary   += "J";
                } else {
                if (!string_at(original, current + 1, 1,
                            ["L","T","K","S","N","M","B","Z"])
                    && !string_at(original, current - 1, 1,
                                ["S","K","L"])) {
                    primary   += "J";
                }
                }
            }
            }

            if (original.substr(current + 1, 1) == 'J') // it could happen
            current += 2;
            else 
            current += 1;
            break;

        case 'K':
            if (original.substr(current + 1, 1) == 'K')
            current += 2;
            else
            current += 1;
            primary   += "K";
            break;

        case 'L':
            if (original.substr(current + 1, 1) == 'L') {
            // spanish e.g. 'cabrillo', 'gallegos'
            if (((current == (length - 3))
                    && string_at(original, current - 1, 4,
                            ["ILLO","ILLA","ALLE"]))
                || ((string_at(original, last-1, 2,
                            ["AS","OS"])
                    || string_at(original, last, 1,
                            ["A","O"]))
                    && string_at(original, current - 1, 4,
                            ["ALLE"]))) {
                primary   += "L";
                current += 2;
                break;
            }
            current += 2;
            } else 
            current += 1;
            primary   += "L";
            break;

        case 'M':
            if ((string_at(original, current - 1, 3,
                        ["UMB"])
                && (((current + 1) == last)
                    || string_at(original, current + 2, 2,
                            ["ER"])))
                // 'dumb', 'thumb'
                || (original.substr(current + 1, 1) == 'M')) {
                current += 2;
            } else {
                current += 1;
            }
            primary   += "M";
            break;

        case 'N':
            if (original.substr(current + 1, 1) == 'N') 
            current += 2;
            else
            current += 1;
            primary   += "N";
            break;

        case 'Ñ':
            current += 1;
            primary   += "N";
            break;

        case 'P':
            if (original.substr(current + 1, 1) == 'H') {
            current += 2;
            primary   += "F";
            break;
            }

            // also account for "campbell" and "raspberry"
            if (string_at(original, current + 1, 1,
                        ["P","B"]))
            current += 2;
            else
            current += 1;
            primary   += "P";
            break;

        case 'Q':
            if (original.substr(current + 1, 1) == 'Q') 
            current += 2;
            else 
            current += 1;
            primary   += "K";
            break;

        case 'R':
            // french e.g. 'rogier', but exclude 'hochmeier'
            if ((current == last)
                && !Slavo_Germanic(original)
                && string_at(original, current - 2, 2,
                        ["IE"])
                && !string_at(original, current - 4, 2,
                            ["ME","MA"])) {
            primary   += "";
            } else {
            primary   += "R";
            }
            if (original.substr(current + 1, 1) == 'R') 
            current += 2;
            else
            current += 1;
            break;

        case 'S':
            // special cases 'island', 'isle', 'carlisle', 'carlysle'
            if (string_at(original, current - 1, 3,
                        ["ISL","YSL"])) {
            current += 1;
            break;
            }

            // special case 'sugar-'
            if ((current == 0)
                && string_at(original, current, 5,
                        ["SUGAR"])) {
            primary   += "X";
            current += 1;
            break;
            }

            if (string_at(original, current, 2,
                        ["SH"])) {
            // germanic
            if (string_at(original, current + 1, 4,
                        ["HEIM","HOEK","HOLM","HOLZ"])) {
                primary   += "S";
            } else {
                primary   += "X";
            }
            current += 2;
            break;
            }

            // italian & armenian 
            if (string_at(original, current, 3,
                        ["SIO","SIA"])
                || string_at(original, current, 4,
                        ["SIAN"])) {
            if (!Slavo_Germanic(original)) {
                primary   += "S";
            } else {
                primary   += "S";
            }
            current += 3;
            break;
            }

            // german & anglicisations, e.g. 'smith' match 'schmidt', 'snider' match 'schneider'
            // also, -sz- in slavic language altho in hungarian it is pronounced 's'
            if (((current == 0)
                && string_at(original, current + 1, 1,
                            ["M","N","L","W"]))
                || string_at(original, current + 1, 1,
                        ["Z"])) {
            primary   += "S";
            if (string_at(original, current + 1, 1,
                        ["Z"]))
                current += 2;
            else
                current += 1;
            break;
            }

            if (string_at(original, current, 2,
                        ["SC"])) {
            // Schlesinger's rule 
            if (original.substr(current + 2, 1) == 'H')
                // dutch origin, e.g. 'school', 'schooner'
                if (string_at(original, current + 3, 2,
                            ["OO","ER","EN","UY","ED","EM"])) {
                // 'schermerhorn', 'schenker' 
                if (string_at(original, current + 3, 2,
                            ["ER","EN"])) {
                    primary   += "X";
                } else {
                    primary   += "SK";
                }
                current += 3;
                break;
                } else {
                if ((current == 0) 
                    && !is_vowel(original, 3)
                    && (original.substr(current + 3, 1) != 'W')) {
                    primary   += "X";
                } else {
                    primary   += "X";
                }
                current += 3;
                break;
                }

                if (string_at(original, current + 2, 1,
                            ["I","E","Y"])) {
                primary   += "S";
                current += 3;
                break;
                }

            // else
            primary   += "SK";
            current += 3;
            break;
            }

            // french e.g. 'resnais', 'artois'
            if ((current == last)
                && string_at(original, current - 2, 2,
                        ["AI","OI"])) {
            primary   += "";
            } else {
            primary   += "S";
            }

            if (string_at(original, current + 1, 1,
                        ["S","Z"]))
            current += 2;
            else 
            current += 1;
            break;

        case 'T':
            if (string_at(original, current, 4,
                        ["TION"])) {
            primary   += "X";
            current += 3;
            break;
            }

            if (string_at(original, current, 3,
                        ["TIA","TCH"])) {
            primary   += "X";
            current += 3;
            break;
            }

            if (string_at(original, current, 2,
                        ["TH"])
                || string_at(original, current, 3,
                            ["TTH"])) {
            // special case 'thomas', 'thames' or germanic
            if (string_at(original, current + 2, 2,
                        ["OM","AM"])
                || string_at(original, 0, 4, ["VAN ","VON "])
                || string_at(original, 0, 3, ["SCH"])) {
                primary   += "T";
            } else {
                primary   += "0";
            }
            current += 2;
            break;
            }

            if (string_at(original, current + 1, 1,
                        ["T","D"]))
            current += 2;
            else
            current += 1;
            primary   += "T";
            break;

        case 'V':
            if (original.substr(current + 1, 1) == 'V')
            current += 2;
            else
            current += 1;
            primary   += "F";
            break;

        case 'W':
            // can also be in middle of word
            if (string_at(original, current, 2, ["WR"])) {
            primary   += "R";
            current += 2;
            break;
            }

            if ((current == 0)
                && (is_vowel(original, current + 1)
                    || string_at(original, current, 2, 
                            ["WH"]))) {
            // Wasserman should match Vasserman 
            if (is_vowel(original, current + 1)) {
                primary   += "A";
            } else {
                // need Uomo to match Womo 
                primary   += "A";
            }
            }

            // Arnow should match Arnoff
            if (((current == last) 
                && is_vowel(original, current - 1))
                || string_at(original, current - 1, 5,
                        ["EWSKI","EWSKY","OWSKI","OWSKY"])
                || string_at(original, 0, 3, ["SCH"])) {
            primary   += "";
            current += 1;
            break;
            }

            // polish e.g. 'filipowicz'
            if (string_at(original, current, 4,
                        ["WICZ","WITZ"])) {
            primary   += "TS";
            current += 4;
            break;
            }

            // else skip it
            current += 1;
            break;

        case 'X':
            // french e.g. breaux 
            if (!((current == last)
                && (string_at(original, current - 3, 3,
                            ["IAU", "EAU"])
                    || string_at(original, current - 2, 2,
                            ["AU", "OU"])))) {
            primary   += "KS";
            }

            if (string_at(original, current + 1, 1,
                        ["C","X"]))
            current += 2;
            else
            current += 1;
            break;

        case 'Z':
            // chinese pinyin e.g. 'zhao' 
            if (original.substr(current + 1, 1) == "H") {
            primary   += "J";
            current += 2;
            break;
            } else if (string_at(original, current + 1, 2,
                            ["ZO", "ZI", "ZA"])
                    || (Slavo_Germanic(original)
                        && ((current > 0)
                            && original.substr(current - 1, 1) != 'T'))) {
            primary   += "S";
            } else {
            primary   += "S";
            }

            if (original.substr(current + 1, 1) == 'Z')
            current += 2;
            else
            current += 1;
            break;

        default:
            current += 1;

        } // end switch

    // printf("<br>ORIGINAL:    '%s'\n", original);
    // printf("<br>current:    '%s'\n", current);
    // printf("<br>  PRIMARY:   '%s'\n", primary);
    // printf("<br>  SECONDARY: '%s'\n", secondary);

    } // end while

    primary   = primary.substr(  0, 4);
    
    return primary;
    
} // end of function MetaPhone
    