const topicInput = document.getElementById('topic-input');
const generateBtn = document.getElementById('generate-btn');
const inputContainer = document.getElementById('input-container');
const cardsWrapper = document.getElementById('cards-wrapper');
const cardsContainer = document.getElementById('cards-container');
const loadingIndicator = document.getElementById('loading-indicator');
const resetBtn = document.getElementById('reset-btn');

// Navigation
const cardCounter = document.getElementById('card-counter');

let flashcardsData = [];
let currentIndex = 0;
let isGeneratingMore = false;

generateBtn.addEventListener('click', () => generateCards(false));
topicInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') generateCards(false);
});

resetBtn.addEventListener('click', resetApp);

function resetApp() {
    // Reset UI state
    cardsWrapper.classList.add('hidden', 'opacity-0');
    resetBtn.classList.add('hidden');
    inputContainer.classList.remove('hidden', '-translate-y-[10vh]', 'scale-90');
    topicInput.value = '';
    topicInput.focus();
    cardsContainer.innerHTML = '';
    flashcardsData = [];
    currentIndex = 0;
}

function showErrorScreen(message, isAppend = false) {
    cardsContainer.innerHTML = '';
    
    const errorCard = document.createElement('div');
    errorCard.className = 'w-full h-full bg-slate-800 rounded-2xl p-8 flex flex-col items-center justify-center text-center ring-1 ring-red-500/20 shadow-2xl';
    errorCard.innerHTML = `
        <div class="mb-6 bg-red-500/10 p-4 rounded-full">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
        </div>
        <h3 class="text-2xl md:text-3xl font-bold text-white mb-3">Oops! Something went wrong</h3>
        <p class="text-slate-400 mb-8 max-w-md text-base md:text-lg">${message}</p>
        
        <div class="flex flex-col sm:flex-row gap-4 w-full justify-center">
            ${isAppend ? `
                <button onclick="showEndScreen()" class="bg-slate-700 hover:bg-slate-600 text-white font-semibold py-3 px-8 rounded-full transition-colors flex items-center justify-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                    </svg>
                    Go Back
                </button>
            ` : `
                <button onclick="resetApp()" class="bg-brand-600 hover:bg-brand-500 text-white font-semibold py-3 px-8 rounded-full transition-colors flex items-center justify-center gap-2 shadow-lg">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                    </svg>
                    Return Home
                </button>
            `}
        </div>
    `;
    
    cardsContainer.appendChild(errorCard);
    
    if (!isAppend) {
        // Show the cards wrapper if it was hidden
        if (cardsWrapper.classList.contains('hidden')) {
            inputContainer.classList.add('hidden');
            cardsWrapper.classList.remove('hidden');
            setTimeout(() => {
                cardsWrapper.classList.remove('opacity-0');
            }, 50);
            resetBtn.classList.remove('hidden');
        }
    }
}

async function generateCards(isAppend = false) {
    const topic = topicInput.value.trim();
    if (!topic) return;

    if (isAppend) {
        isGeneratingMore = true;
        // Keep current visual state, maybe show a toast or small indicator?
        // Using valid HTML for the 'generating more' card
        cardsContainer.innerHTML = `
            <div class="w-full h-full bg-slate-800 rounded-2xl p-8 flex flex-col items-center justify-center text-center ring-1 ring-white/10 animate-pulse">
                <div class="text-brand-400 mb-4">
                    <svg class="animate-spin h-10 w-10 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </div>
                <h3 class="text-xl font-bold text-white">Expanding your knowledge...</h3>
            </div>
        `;
    } else {
        generateBtn.disabled = true;
        // Show loading initially
        loadingIndicator.classList.remove('hidden');
    }

    try {
        const payload = { topic };

        // If appending, send existing cards context to avoid duplicates
        if (isAppend) {
            payload.context = flashcardsData.map(c => c.front);
        }

        const response = await fetch('/api/generate-cards', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Failed to generate cards');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let isFirstCard = true;
        let newCardsCount = 0;

        // Loop to read the stream
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process chunks
            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, newlineIndex).trim();
                buffer = buffer.slice(newlineIndex + 1);

                if (line) {
                    try {
                        const card = JSON.parse(line);

                        // Add to data
                        flashcardsData.push(card);
                        newCardsCount++;

                        // Determine behavior based on state
                        if (isAppend) {
                            // If appending, we might already be on the "End" screen or viewing the last card
                            // If we collected a new card, we can show it immediately if we were on the loader
                            if (newCardsCount === 1) {
                                // We were showing the "Expanding..." loader at currentIndex
                                // currentIndex is likely where the new card should be
                                // Actually, if we appended, currentIndex was at data.length (end screen)
                                // So renderCurrentCard should now show this new card
                                renderCurrentCard();
                            } else {
                                // Background update of counters
                                cardCounter.textContent = `${currentIndex + 1} / ${flashcardsData.length}`;
                            }
                        } else {
                            // New generation
                            if (isFirstCard) {
                                isFirstCard = false;
                                currentIndex = 0;
                                loadingIndicator.classList.add('hidden');
                                inputContainer.classList.add('hidden');
                                cardsWrapper.classList.remove('hidden');
                                setTimeout(() => {
                                    cardsWrapper.classList.remove('opacity-0');
                                }, 50);
                                resetBtn.classList.remove('hidden');
                                // Render the first card immediately
                                renderCurrentCard();
                            } else {
                                // Just update the counter usually, or let the user navigate to it
                                // If the user is staring at card 1, update counter to "1 / X"
                                cardCounter.textContent = `${currentIndex + 1} / ${flashcardsData.length}`;
                            }
                        }

                    } catch (e) {
                        console.warn('Error parsing JSON line from stream:', e);
                    }
                }
            }
        }

        if (newCardsCount === 0 && !isAppend) {
            throw new Error('No cards generated');
        }

    } catch (error) {
        console.error(error);

        if (isAppend) {
            showErrorScreen('Could not generate more cards. Please try again later.', true);
        } else {
            showErrorScreen('Unable to generate flashcards. Please check your connection and try again.', false);
            loadingIndicator.classList.add('hidden');
            generateBtn.disabled = false;
        }
    } finally {
        generateBtn.disabled = false;
        loadingIndicator.classList.add('hidden');
        isGeneratingMore = false;
    }
}

function navigateCard(direction) {
    if (isGeneratingMore) return;

    const newIndex = currentIndex + direction;

    // Allow going one step past the last card to show the "End/More" screen
    if (newIndex >= 0 && newIndex <= flashcardsData.length) {
        currentIndex = newIndex;
        if (currentIndex === flashcardsData.length) {
            showEndScreen();
        } else {
            renderCurrentCard();
        }
    }
}

function showEndScreen() {
    // Update Counter
    cardCounter.textContent = `COMPLETED`;

    cardsContainer.innerHTML = '';

    const endCard = document.createElement('div');
    endCard.className = 'w-full h-full bg-slate-800 rounded-2xl p-8 flex flex-col items-center justify-center text-center ring-1 ring-white/10 shadow-2xl';
    endCard.innerHTML = `
        <div class="mb-6 bg-brand-500/10 p-4 rounded-full">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
        </div>
        <h3 class="text-3xl font-bold text-white mb-2">Topic Completed!</h3>
        <p class="text-slate-400 mb-8 max-w-sm">You've mastered this set. Ready to dive deeper or start something new?</p>
        
        <div class="flex flex-col sm:flex-row gap-4 w-full justify-center">
            <button onclick="generateCards(true)" class="bg-brand-600 hover:bg-brand-500 text-white font-semibold py-3 px-8 rounded-full transition-colors flex items-center justify-center gap-2 shadow-lg">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
                Load More
            </button>
            <button onclick="resetApp()" class="bg-slate-700 hover:bg-slate-600 text-white font-semibold py-3 px-8 rounded-full transition-colors flex items-center justify-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                New Topic
            </button>
        </div>
    `;

    cardsContainer.appendChild(endCard);
}

// Keyboard Navigation
document.addEventListener('keydown', (e) => {
    // Only work if cards are visible and not hidden
    if (cardsWrapper.classList.contains('hidden')) return;

    // Don't interfere if user is typing in an input (though input is hidden now, good practice)
    if (document.activeElement.tagName === 'INPUT') return;

    if (e.key === 'ArrowRight') {
        navigateCard(1);
    } else if (e.key === 'ArrowLeft') {
        navigateCard(-1);
    }
});

// Touch swipe navigation (mobile)
let touchStartX = null;
let touchStartY = null;
let touchEndX = null;
let touchEndY = null;
const SWIPE_THRESHOLD = 50; // pixels

cardsContainer.addEventListener('touchstart', (e) => {
    if (!e.touches || e.touches.length === 0) return;
    const t = e.touches[0];
    touchStartX = t.clientX;
    touchStartY = t.clientY;
});

cardsContainer.addEventListener('touchend', (e) => {
    if (!e.changedTouches || e.changedTouches.length === 0) return;
    const t = e.changedTouches[0];
    touchEndX = t.clientX;
    touchEndY = t.clientY;

    const dx = touchEndX - touchStartX;
    const dy = touchEndY - touchStartY;

    // Only consider mostly-horizontal swipes
    if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > SWIPE_THRESHOLD) {
        if (dx < 0) {
            navigateCard(1); // swipe left → next
        } else {
            navigateCard(-1); // swipe right → previous
        }
    }

    touchStartX = touchStartY = touchEndX = touchEndY = null;
});

function flipCurrentCard() {
    const cardElement = cardsContainer.querySelector('.group');
    if (cardElement) cardElement.classList.toggle('rotate-y-180');
}

function renderCurrentCard() {
    if (flashcardsData.length === 0) return;

    const card = flashcardsData[currentIndex];

    // Update Counter
    cardCounter.textContent = `${currentIndex + 1} / ${flashcardsData.length}`;

    cardsContainer.innerHTML = '';

    const cardElement = document.createElement('div');
    // Perspective wrapper handled by parent, this is the card itself
    cardElement.className = 'group relative w-full h-full cursor-pointer duration-500 transform-style-3d';

    // Helper to stop flip when clicking specific elements
    const handleNextClick = (e) => {
        e.stopPropagation();
        navigateCard(1);
    };

    // Helper to escape HTML for safe rendering
    const escapeHtml = (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };

    // Helper to convert newlines to paragraphs
    const formatParagraphs = (text) => {
        // Split by newlines and filter out empty strings
        const paragraphs = text.split('\n').filter(p => p.trim() !== '');

        // If only one paragraph, return it as a single <p>
        if (paragraphs.length === 1) {
            return `<p class="text-lg md:text-2xl text-white leading-relaxed font-medium select-none">${escapeHtml(paragraphs[0])}</p>`;
        }

        // Multiple paragraphs: wrap each in a <p> tag with spacing
        return paragraphs.map(p =>
            `<p class="text-lg md:text-2xl text-white leading-relaxed font-medium select-none mb-3 md:mb-4 last:mb-0">${escapeHtml(p.trim())}</p>`
        ).join('');
    };

    // Helper to stop flip when clicking specific elements
    const stopPropagation = (e) => {
        e.stopPropagation();
    };

    // Build code snippet HTML if present
    let codeSnippetHtml = '';
    if (card.code && card.code.trim() !== '') {
        codeSnippetHtml = `
            <div class="code-wrapper w-full mt-6 text-left relative group/code cursor-pointer">
                <!-- Code Preview -->
                <div class="code-container relative max-h-36 overflow-hidden bg-slate-900/80 rounded-lg ring-1 ring-white/5 transition-all duration-300 group-hover/code:ring-brand-500/30 group-hover/code:shadow-lg group-hover/code:shadow-brand-500/10">
                    <pre class="p-4 text-sm font-mono text-emerald-300 whitespace-pre-wrap break-words opacity-80 group-hover/code:opacity-30 transition-opacity duration-300"><code>${escapeHtml(card.code)}</code></pre>
                </div>
                
                <!-- Centered Hover Overlay -->
                <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover/code:opacity-100 transition-opacity duration-200 pointer-events-none">
                    <button class="expand-code-btn bg-brand-600 hover:bg-brand-500 text-white text-sm font-bold py-2 px-6 rounded-full shadow-xl ring-1 ring-white/20 transform scale-95 group-hover/code:scale-105 transition-all duration-200 flex items-center gap-2 pointer-events-auto">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" /></svg>
                        Expand
                    </button>
                </div>
            </div>
        `;
    }

    cardElement.innerHTML = `
            <!-- Front -->
            <div class="absolute w-full h-full bg-slate-800 rounded-2xl p-5 md:p-10 flex flex-col items-center justify-center text-center backface-hidden ring-1 ring-white/10 shadow-2xl">
                 <!-- Main Content -->
                 <div class="flex-grow flex items-center justify-center">
                    <h3 class="text-3xl md:text-5xl font-bold text-white leading-tight select-none">${card.front}</h3>
                 </div>
                 <!-- Footer -->
                 <p class="text-slate-500 text-xs uppercase tracking-widest mt-4">Click to flip</p>
            </div>
            
            <!-- Back -->
            <div class="card-back absolute w-full h-full bg-gradient-to-br from-brand-900 to-slate-900 rounded-2xl p-5 md:p-10 flex flex-col items-start justify-start text-center backface-hidden rotate-y-180 ring-1 ring-white/10 shadow-2xl overflow-hidden">
                 <div class="flex-grow flex flex-col items-center justify-start w-full transform rotate-[0deg] overflow-auto hide-scrollbar pt-2"> 
                    <div class="text-left w-full">
                        ${formatParagraphs(card.back)}
                    </div>
                    ${codeSnippetHtml}
                </div>
                 
                 <!-- On-card Next Button for seamless mouse usage -->
                 <button class="next-card-btn mt-4 md:mt-6 bg-white/10 hover:bg-white/20 text-white text-base md:text-base font-semibold py-3 md:py-4 px-6 md:px-10 rounded-full transition-all flex items-center gap-2 backdrop-blur-sm shadow-lg ring-1 ring-white/10 group-hover:bg-brand-500 group-hover:shadow-brand-500/50 shrink-0 self-center">
                    Next Card <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 md:h-5 md:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
                 </button>
            </div>
    `;

    // Click to flip (main handler)
    cardElement.onclick = function () {
        this.classList.toggle('rotate-y-180');
    };

    // Attach specific handler for the button inside
    const nextBtnInternal = cardElement.querySelector('.next-card-btn');
    if (nextBtnInternal) nextBtnInternal.onclick = handleNextClick;

    // Attach handlers for code expansion
    const expandBtn = cardElement.querySelector('.expand-code-btn');
    const cardBack = cardElement.querySelector('.card-back');

    if (expandBtn && cardBack) {
        expandBtn.onclick = (e) => {
            e.stopPropagation(); // Prevent flip

            // Create full-card code overlay
            const codeOverlay = document.createElement('div');
            codeOverlay.className = 'code-fullscreen absolute inset-0 z-50 bg-slate-900 rounded-2xl p-6 flex flex-col text-left';
            codeOverlay.innerHTML = `
                <div class="flex justify-between items-center mb-4">
                    <span class="text-slate-400 text-sm font-medium uppercase tracking-wide">Code Snippet</span>
                    <button class="close-code-btn bg-slate-700 hover:bg-slate-600 text-white text-sm font-semibold py-2 px-4 rounded-full shadow-lg ring-1 ring-white/10 transition-all flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
                        Close
                    </button>
                </div>
                <div class="flex-grow overflow-auto rounded-lg bg-slate-800/50 ring-1 ring-white/5">
                    <pre class="p-4 text-sm font-mono text-emerald-300 whitespace-pre-wrap break-words h-full"><code>${escapeHtml(card.code)}</code></pre>
                </div>
            `;

            // Add close handler
            const closeBtn = codeOverlay.querySelector('.close-code-btn');
            closeBtn.onclick = (e) => {
                e.stopPropagation();
                codeOverlay.remove();
            };

            // Prevent flip when clicking overlay
            codeOverlay.onclick = (e) => e.stopPropagation();

            // Add to card back
            cardBack.appendChild(codeOverlay);
        };

        // Prevent flip when clicking on the code container
        const codeContainer = cardElement.querySelector('.code-container');
        if (codeContainer) codeContainer.onclick = stopPropagation;
        expandBtn.onclick && (expandBtn.parentElement.onclick = stopPropagation);
    }

    cardsContainer.appendChild(cardElement);
}
