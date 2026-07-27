// lib/errorDiagnoser.ts
import errorPatternsRaw from '../data/error_patterns.json';

export interface ErrorPattern {
    pattern: string;
    arti: string;
    penyebab: string;
    solusi: string;
}

interface CompiledPattern extends ErrorPattern {
    regex: RegExp;
}

// This runs ONCE when the module is first imported anywhere in your app
// (e.g. when the page/component that uses it first loads)
const compiledPatterns: CompiledPattern[] = (errorPatternsRaw as ErrorPattern[]).map(
    (item) => ({
        ...item,
        regex: new RegExp(item.pattern, 'i'),
    })
);

export function diagnosaError(tracebackText: string): ErrorPattern | null {
    for (const item of compiledPatterns) {
        if (item.regex.test(tracebackText)) {
            return item;
        }
    }
    return null;
}