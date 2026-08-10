# Food Inspection AI Dashboard — Design Brainstorming

## Three Stylistic Approaches

**Approach A: Industrial Precision** (probability: 0.07)
Clean, high-contrast monochrome base with amber/orange accents inspired by industrial HMI (Human-Machine Interface) panels. Tight grid layouts, monospaced data readouts, and hard-edged components that feel like a real factory control system.

**Approach B: Scientific Lab Dashboard** (probability: 0.05)
Deep navy and slate backgrounds with cyan/teal data highlights. Inspired by scientific instrument software — precise, data-dense, with subtle grid lines and chart-forward layouts. Feels like a professional laboratory information management system.

**Approach C: Modern Ops Platform** (probability: 0.08)
Light, airy background with a deep forest-green primary and warm cream/sand accents. Inspired by modern DevOps/MLOps platforms (Grafana, Weights & Biases). Clean sidebar navigation, generous whitespace, and card-based data presentation.

---

## Selected Approach: **A — Industrial Precision**

This is a food quality control system used on production lines. The UI must feel authoritative, precise, and immediately readable under factory lighting conditions. Industrial Precision is the only aesthetic that earns trust in this context.

### Design Movement
Industrial HMI (Human-Machine Interface) meets modern data engineering. Reference: Siemens SIMATIC, Rockwell FactoryTalk, but elevated with modern typography and micro-interactions.

### Core Principles
1. **Data legibility above all** — every number must be instantly readable at a glance
2. **Status-first hierarchy** — inspection status (OK/DEFECT/UNCERTAIN) must dominate visual weight
3. **Precision without clutter** — dense information organized in tight, purposeful grids
4. **Operational trust** — the UI must feel reliable and production-grade, never playful

### Color Philosophy
- **Background:** Near-black slate (`oklch(0.12 0.008 240)`) — reduces eye strain in factory environments
- **Surface:** Dark charcoal (`oklch(0.18 0.008 240)`) for cards and panels
- **Primary Accent:** Amber/yellow (`oklch(0.78 0.18 85)`) — the universal industrial warning/highlight color
- **Status Colors:** Emerald green (OK), Crimson red (DEFECT), Amber (UNCERTAIN), Slate gray (SKIPPED)
- **Text:** Near-white for primary, muted slate for secondary

### Layout Paradigm
Persistent left sidebar (64px collapsed / 240px expanded) with icon + label navigation. Main content area uses asymmetric grid layouts — never centered hero layouts. Dashboard uses a 12-column grid with varying card widths to create visual rhythm.

### Signature Elements
1. **Monospaced data readouts** — confidence scores, timestamps, and bbox coordinates rendered in a monospace font
2. **Status indicator strips** — thin colored left-border on cards indicating inspection status
3. **Scan-line animation** — subtle horizontal scan line animation on the live inspection view

### Interaction Philosophy
Every action has immediate visual feedback. Hover states are subtle but present. Transitions are fast (150-200ms) and functional, never decorative. The UI responds like a real-time system.

### Animation
- Page transitions: 150ms fade-in from slight translateY(8px)
- Card entrances: staggered 40ms delay per card, fade + scale(0.97→1)
- Status badge pulses: slow 2s pulse on DEFECT status
- Scan line: 3s linear loop on inspection view
- All animations respect `prefers-reduced-motion`

### Typography System
- **Display/Headings:** `Space Grotesk` — geometric, technical, authoritative
- **Body/UI:** `Inter` — clean and readable
- **Data/Monospace:** `JetBrains Mono` — for all numeric readouts, coordinates, confidence scores
- Scale: 12px (data labels) → 14px (body) → 16px (subheading) → 24px (heading) → 36px (display)

### Brand Essence
**FoodScan AI** — Real-time quality intelligence for food production lines. For quality engineers and production managers who need zero-tolerance defect detection. Unlike generic dashboards, it speaks the language of the factory floor.
Personality: **Precise. Reliable. Authoritative.**

### Brand Voice
- Headlines: "Zero defects. Real-time." / "Every item. Every frame."
- CTAs: "Run Inspection" / "View Report" / "Analyze Now"
- No filler phrases like "Welcome to our platform"

### Wordmark & Logo
A stylized scan-line icon — a horizontal bar with a thin scanning beam — rendered in amber on dark background. Represents the core action of the system.

### Signature Brand Color
**Amber `oklch(0.78 0.18 85)`** — unmistakably industrial, universally recognized as "attention" in factory contexts.

## Style Decisions
- Dark theme as default (factory floor readability)
- Monospace font for all numeric data
- Status colors: green=#22c55e, red=#ef4444, amber=#f59e0b, gray=#6b7280
- **Panel language:** Industrial Precision panels use hard-edged HMI geometry (border-radius: 2px), visible grid/divider structure (hmi-grid class), and amber schematic accent lines (amber-accent-line class); avoid soft generic SaaS cards.
- **Status hierarchy:** OK / DEFECT / UNCERTAIN / SKIPPED states are the strongest visual hierarchy on every operational screen — metric cards use status color as dominant text color, not just as accent.
- **Brand motif:** FoodScan AI scan-line mark appears in sidebar top accent, active nav states, and key data panel headers via amber left-border strips.
- **Copy voice:** All labels and headings use UPPERCASE MONOSPACE to reinforce factory-floor authority (e.g., "RUN INSPECTION", "FRAME #1000", "API ONLINE").
