// search-v2 experiment: lazy vector loading + int8-quantized vectors.
//
// Mirrors the search behaviour of /assets/semantic_search.js so timings and
// rankings are directly comparable, with two changes:
//   1. Vectors are NOT fetched until explicitly requested (production loads
//      them eagerly on DOMContentLoaded — the v2 flow defers to first search).
//   2. Supports both the fp32 JSON format and the int8-quantized format
//      produced by quantize_embeddings.py ("govdoc-int8-v1").
//
// Exported as a module so benchmark.html can drive it headlessly; the
// production integration path is documented in README.md.

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;

const Q8_FORMAT = 'govdoc-int8-v1';

let extractor = null;

export async function initEmbedder() {
    if (extractor) return extractor;
    // dtype pinned: q8 is the smallest usable variant for this model
    // (118MB int8 + ~32MB tokenizer; q4 is larger for this architecture).
    extractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', { dtype: 'q8' });
    return extractor;
}

export async function embedQuery(text) {
    const model = await initEmbedder();
    const output = await model(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

function b64ToInt8(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const i8 = new Int8Array(u8.length);
    for (let i = 0; i < u8.length; i++) i8[i] = (u8[i] << 24) >> 24; // sign-extend
    return i8;
}

function cosineF32(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot; // both vectors are mean-pooled + normalized already
}

// Cosine on raw int8 vectors. Per-vector scale cancels out of cosine
// (cos(s*a, s*b) === cos(a, b)), so ranking needs no dequantization pass.
// The query vector is quantized with the same symmetric scheme.
function cosineInt8(queryI8, docI8) {
    let dot = 0, qn = 0, dn = 0;
    for (let i = 0; i < queryI8.length; i++) {
        const q = queryI8[i], d = docI8[i];
        dot += q * d;
        qn += q * q;
        dn += d * d;
    }
    const denom = Math.sqrt(qn) * Math.sqrt(dn);
    return denom ? dot / denom : 0;
}

function quantizeQuery(f32) {
    let maxAbs = 0;
    for (let i = 0; i < f32.length; i++) {
        const m = Math.abs(f32[i]);
        if (m > maxAbs) maxAbs = m;
    }
    const i8 = new Int8Array(f32.length);
    if (!maxAbs) return i8;
    const scale = maxAbs / 127;
    for (let i = 0; i < f32.length; i++) {
        i8[i] = Math.max(-127, Math.min(127, Math.round(f32[i] / scale)));
    }
    return i8;
}

// Load a vectors file (fp32 legacy or int8) and normalize entries to
// { id, title, text, vec }. Deferring this call IS the lazy-load change.
export async function loadVectors(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${response.status} ${path}`);
    const data = await response.json();

    if (data && data.format === Q8_FORMAT) {
        return data.entries.map(entry => ({
            id: entry.id,
            title: entry.title ?? entry.id,
            text: entry.body || entry.text || '',
            vec: { kind: 'i8', data: b64ToInt8(entry.embedding) },
        }));
    }
    // Legacy fp32 shape: {embedding: [...] | [[...]]}
    return data.map(entry => ({
        id: entry.id,
        title: entry.title ?? entry.id,
        text: entry.body || entry.text || '',
        vec: {
            kind: 'f32',
            data: (entry.embedding.length === 1 ? entry.embedding[0] : entry.embedding),
        },
    }));
}

// Rank all entries against one query embedding. Returns the same result
// shape as semanticDocSearch() in the production file.
export function rankVectors(queryEmbedding, entries, limit = 25) {
    const useInt8 = entries.length > 0 && entries[0].vec.kind === 'i8';
    const queryVec = useInt8 ? quantizeQuery(queryEmbedding) : null;
    const scored = entries.map(entry => {
        let similarity;
        if (useInt8) {
            similarity = cosineInt8(queryVec, entry.vec.data);
        } else {
            similarity = cosineF32(queryEmbedding, entry.vec.data);
        }
        return { id: entry.id, title: entry.title, text: entry.text, similarity };
    });
    scored.sort((a, b) => b.similarity - a.similarity);
    return scored
        .filter(x => x.similarity >= 0.3)
        .slice(0, limit)
        .map(x => ({
            url: './sources/' + x.id + '/index.html',
            title: x.title,
            similarity: x.similarity,
            channel: 'semantic',
        }));
}