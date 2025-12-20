\# Implementation Specification: Flux-Ready Dynamic Exploration UI



\*\*Context \& Stack:\*\*



\* \*\*Frontend:\*\* Vue.js 3 (Composition API).

\* \*\*UI Framework:\*\* Bootstrap 5.

\* \*\*Backend:\*\* C++ Server (Stable Diffusion).

\* \*\*AI/LLM:\*\* Real-time calls to an LLM endpoint (e.g., via generic HTTP wrapper to OpenAI/Ollama API) for prompt engineering.



\## 1. Core Feature: Dynamic Exploration Grid



A 3x3 grid where the center image is the "Anchor". The 8 surrounding cells are dynamically generated variations.



\*\*Key Requirements:\*\*



1\. \*\*Low Step Count Support:\*\* Must support Steps as low as 1-4 for Flux/Turbo models.

2\. \*\*Parameter Locking:\*\* Users must be able to lock specific parameters (Steps, Scheduler, Guidance, Prompt). Locked parameters \*\*never\*\* change in the variations.

3\. \*\*Dynamic Slots:\*\* The 8 variation slots are not hardcoded. They are filled based on what is unlocked.

4\. \*\*Real LLM Integration:\*\* Prompt variations use an actual LLM to rewrite text, not placeholders.



\## 2. Data Structures



\### 2.1 The Parameter Object



```typescript

interface GenParams {

&nbsp; prompt: string;

&nbsp; seed: number;

&nbsp; steps: number;        // Allow low values (e.g., 4)

&nbsp; guidanceScale: number; // Flux often uses 1.5 - 3.5, SD uses 7+

&nbsp; scheduler: string;

&nbsp; width: number;

&nbsp; height: number;

}



```



\### 2.2 The Lock Mask



```typescript

interface LockedParams {

&nbsp; prompt: boolean;

&nbsp; steps: boolean;

&nbsp; guidance: boolean;

&nbsp; scheduler: boolean;

&nbsp; seed: boolean; // Usually unlocked, but maybe user wants to test params on same seed

}



```



\## 3. Architecture: The Mutation Engine



\*\*Logic Flow:\*\*

When `generateVariations(centerParams, locks)` is called:



1\. \*\*Identify Opportunities:\*\* Create a list of possible mutations based on `!locks`.

\* \*Prompt Unlocked:\* Add `Prompt:Creative`, `Prompt:Subtle`, `Prompt:Shorten`.

\* \*Steps Unlocked:\* Add `Steps:Lower` (min 3), `Steps:Higher`.

\* \*Guidance Unlocked:\* Add `Guidance:Lower`, `Guidance:Higher`.

\* \*Scheduler Unlocked:\* Add `Scheduler:Next`.





2\. \*\*Fill Slots (Priority Queue):\*\*

\* We have exactly 8 slots.

\* \*\*Rule 1:\*\* If `Seed` is unlocked, always reserve at least 1 slot for `RandomSeed`.

\* \*\*Rule 2:\*\* If `Prompt` is unlocked, prioritize 2-3 distinct LLM variations.

\* \*\*Rule 3:\*\* Fill remaining slots with `RandomSeed` (if allowed) or intensify other variations (e.g., if only Steps are unlocked, do -1, -2, +1, +2).





3\. \*\*Execution:\*\* Return array of 8 `GenParams`.



\## 4. LLM Service Specification (`services/llm.ts`)



Must perform a real API call (mock the endpoint URL config for now, but implement the fetch logic).



\*\*Function:\*\* `rewritePrompt(originalPrompt: string, style: 'creative' | 'subtle' | 'detail'): Promise<string>`



\*\*System Prompt Strategy:\*\*



> "You are a Stable Diffusion prompt expert.

> User Prompt: '{style}'.

> Constraint: Keep the main subject intact. Output ONLY the raw prompt text, no markdown."



\## 5. UI Implementation (Vue + Bootstrap)



\### 5.1 The Control Bar (Locking Mechanism)



Above the grid, render small "Lock" toggles (Icons: Open Padlock / Closed Padlock) for each parameter category.



\* `\[🔓 Prompt]` `\[🔒 Steps]` `\[🔓 Guidance]` `\[🔓 Scheduler]`



\### 5.2 The Grid Components



\* \*\*Center Card:\*\* Highlighted border. Shows current params.

\* \*\*Neighbor Cards:\*\*

\* \*\*Badge:\*\* Top-right corner badge indicating the mutation type (e.g., "LLM: Creative", "Steps -2", "New Seed").

\* \*\*Interaction:\*\*

\* \*\*Click:\*\* Promotes to Center.

\* \*\*Right-Click/Long-Press:\*\* "Ban this path" (Optional, maybe for later).





\* \*\*Edit Button:\*\* Opens specific params in Advanced Editor.







\## 6. Implementation Tasks for AI Agent



\*\*Phase 1: Logic \& State (Pinia)\*\*



1\. Create `ExplorationStore`. State: `centerParams`, `locks`, `neighborCells`.

2\. Implement `MutationBuilder` class.

\* \*Crucial:\* Handle the "Steps" math carefully. If current steps = 4, `Steps:Lower` should probably stay 4 or go to 3, never 0.

\* Implement the slot-filling algorithm described in Section 3.







\*\*Phase 2: LLM Service\*\*



1\. Create `LLMService` with a `rewrite` method.

2\. Use a simple `fetch` to a configurable endpoint (default to `http://localhost:11434/api/generate` for Ollama compatibility or similar).



\*\*Phase 3: The View\*\*



1\. Build the 3x3 Grid using Bootstrap rows/cols.

2\. Add the Locking Toggles component.

3\. Connect the `onCellClick` -> `updateCenter` -> `generateVariations` loop.



---



\### Prompt for the Coding Agent



Use this prompt to start the implementation:



> "I need to implement the \*\*Dynamic Mutation Engine\*\* and \*\*LLM Service\*\* for my Stable Diffusion Vue/Bootstrap app.

> \*\*Requirements:\*\*

> 1. Create a `MutationBuilder` class (TypeScript). It takes `currentParams` and `lockedFields` (boolean map).

> 2. It returns exactly 8 variation objects.

> 3. \*\*Logic:\*\* It dynamically decides which parameters to vary based on what is NOT locked.

> \* If \*\*Prompt\*\* is unlocked, it calls `LLMService` (async) for variations.

> \* If \*\*Steps\*\* is unlocked, vary steps (respecting min=3).

> \* Fill empty slots with Random Seeds.

> 

> 

> 4. Create the `LLMService` that sends the prompt to a generic API endpoint expecting a JSON response.

> 

> 

> Please write the code for `MutationBuilder.ts` and `LLMService.ts` following the specification above."

