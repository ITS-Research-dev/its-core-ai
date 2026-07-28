'use client';

import { useState } from 'react';
import { diagnosaError, type ErrorPattern } from '@/lib/errorDiagnoser';

export default function ErrorChecker() {
    const [result, setResult] = useState<ErrorPattern | null>(null);

    function handleCheck(traceback: string) {
        // Just calls the function — no re-reading, no re-compiling
        const diagnosis = diagnosaError(traceback);
        setResult(diagnosis);
    }

    return (
        <div>
            <button onClick={() => handleCheck('ZeroDivisionError: division by zero')}>
                Check
            </button>
            {result && (
                <div>
                    <p><b>Arti:</b> {result.arti}</p>
                    <p><b>Penyebab:</b> {result.penyebab}</p>
                    <p><b>Solusi:</b> {result.solusi}</p>
                </div>
            )}
        </div>
    );
}