# Design System

UI/UX design system for StoMar. Dark-first, professional quant terminal aesthetic.

---

## Design Principles

1. **Dark-first:** Deep navy/slate base, light mode is secondary
2. **Data-dense:** Maximize information per screen, minimize whitespace
3. **Professional:** Bloomberg Terminal meets modern web design
4. **Consistent:** Unified color palette, typography, spacing
5. **Accessible:** High contrast ratios, clear visual hierarchy

---

## Color Palette

### Brand Colors
| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Cyan | `#22d3ee` | 34, 211, 238 | Primary accent, links, highlights |
| Blue | `#3b82f6` | 59, 130, 246 | Secondary accent, gradients |
| Violet | `#8b5cf6` | 139, 92, 246 | Tertiary accent, gradients |

### Semantic Colors
| Name | Hex | Usage |
|------|-----|-------|
| Emerald | `#10b981` | Profit, BUY signals, Bull regime, positive values |
| Rose | `#f43f5e` | Loss, SELL signals, Bear regime, negative values |
| Amber | `#f59e0b` | Warnings, Neutral regime, caution states |
| Slate | `#94a3b8` | Secondary text, muted labels |

### Background Colors
| Name | Value | Usage |
|------|-------|-------|
| Primary | `#0a0e17` | App background |
| Secondary | `#111827` | Card backgrounds, tab bar |
| Card | `rgba(17, 24, 39, 0.85)` | Glass cards |
| Card Hover | `rgba(30, 41, 59, 0.9)` | Card hover state |

### Border Colors
| Name | Value | Usage |
|------|-------|-------|
| Primary | `rgba(51, 65, 85, 0.5)` | Default borders |
| Hover | `rgba(99, 179, 237, 0.4)` | Hover state borders |
| Glow | `rgba(99, 179, 237, 0.2)` | Glow effects |

---

## Typography

### Font Families
| Family | Weight | Usage |
|--------|--------|-------|
| Inter | 300-900 | Primary UI font (headings, body, labels) |
| JetBrains Mono | 400-600 | Monospace for data, numbers, code |

### Type Scale
| Name | Size | Weight | Letter-spacing | Usage |
|------|------|--------|----------------|-------|
| Hero | 2.2rem | 800 | -0.03em | Main section titles |
| Heading | 1.8rem | 800 | -0.02em | Tab headers |
| Subheading | 1.3rem | 700 | -0.01em | Sub-section titles |
| Metric Large | 1.75rem | 800 | -0.03em | KPI values |
| Metric Medium | 1.2rem | 700 | -0.01em | Secondary KPI values |
| Body | 0.85rem | 400 | 0 | Default text |
| Caption | 0.75rem | 500 | 0.01em | Helper text |
| Label | 0.7rem | 600 | 0.08em | Uppercase labels |
| Micro | 0.65rem | 500 | 0.05em | Fine print |

---

## Spacing System

| Token | Value | Usage |
|-------|-------|-------|
| xs | 0.25rem | Tight gaps (inline elements) |
| sm | 0.5rem | Small gaps (between related items) |
| md | 0.75rem | Medium gaps (card internals) |
| lg | 1rem | Large gaps (between sections) |
| xl | 1.5rem | Extra large (section margins) |
| 2xl | 2rem | Page-level spacing |

---

## Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| sm | 8px | Tabs, small elements |
| md | 12px | Cards, inputs, buttons |
| lg | 16px | Large cards, glass panels |
| xl | 24px | Full-width containers |
| pill | 100px | Tags, badges, prediction pills |

---

## Component Library

### Glass Card (`.glass`)
```css
/* Primary container for all content blocks */
background: var(--bg-card);          /* rgba(17,24,39,0.85) */
backdrop-filter: blur(20px);
border: 1px solid var(--border-primary);
border-radius: 16px;
padding: 1.5rem;
transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
```
**Hover:** Border turns cyan, subtle glow shadow, slight upward lift.

### Stat Item (`.stat-item`)
```css
/* Metric display card — the most used component */
background: var(--bg-card);
border: 1px solid var(--border-primary);
border-radius: 12px;
padding: 1rem;
text-align: center;
transition: all 0.2s ease;
```
**Structure:**
```
┌─────────────────────┐
│   .metric-label     │  (uppercase, 0.7rem, muted)
│   .metric-val       │  (1.75rem, mono, bold)
│   [delta arrow]     │  (optional, green/red)
│   .metric-sub       │  (optional, 0.8rem, secondary)
└─────────────────────┘
```

### Tag / Badge (`.tag`)
```css
/* Inline status indicator */
display: inline-flex;
align-items: center;
gap: 0.3rem;
padding: 0.25rem 0.75rem;
border-radius: 100px;
font-size: 0.7rem;
font-weight: 600;
text-transform: uppercase;
```
**Variants:**
| Class | Color | Usage |
|-------|-------|-------|
| `.tag-buy` | Emerald | BUY signals |
| `.tag-sell` | Rose | SELL signals |
| `.tag-neutral` | Slate | Neutral states |
| `.tag-up` | Emerald | Positive movement |
| `.tag-down` | Rose | Negative movement |
| `.tag-warn` | Amber | Warnings |
| `.tag-info` | Blue | Information |

### Prediction Pill (`.pred-pill`)
```css
/* Large centered BUY/SELL indicator */
display: inline-flex;
align-items: center;
gap: 0.5rem;
padding: 0.5rem 1.5rem;
border-radius: 100px;
font-weight: 700;
font-size: 1.1rem;
```
**Variants:**
| Class | Style |
|-------|-------|
| `.pred-pill.up` | Green bg, green border, green glow |
| `.pred-pill.down` | Rose bg, rose border, rose glow |

### Section Header (`.section-header`)
```css
/* Section divider with gradient bar */
font-size: 0.7rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.12em;
color: var(--text-muted);
border-bottom: 1px solid var(--border-primary);
```
**Includes:** A 3px gradient bar on the left via `::before` pseudo-element.

### Gradient Text (`.gradient-text`)
```css
/* Brand gradient text effect */
background: linear-gradient(135deg, #22d3ee, #3b82f6, #8b5cf6);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
font-weight: 800;
```

### Regime Badge (`.regime-badge`)
```css
/* Market regime indicator */
display: inline-flex;
align-items: center;
gap: 0.4rem;
padding: 0.4rem 1rem;
border-radius: 100px;
font-size: 0.8rem;
font-weight: 700;
```
**Variants:**
| Class | Color |
|-------|-------|
| `.regime-bull` | Emerald |
| `.regime-bear` | Rose |
| `.regime-neutral` | Amber |

### Live Dot (`.live-dot`)
```css
/* Animated market status indicator */
width: 6px;
height: 6px;
border-radius: 50%;
animation: pulse 2s ease-in-out infinite;
```
**Variants:**
| Class | Color |
|-------|-------|
| `.live-dot.open` | Emerald with glow |
| `.live-dot.closed` | Rose with glow |

---

## Animations

| Name | CSS | Duration | Usage |
|------|-----|----------|-------|
| `fadeIn` | `opacity: 0 + translateY(8px)` → visible | 0.4s ease-out | Card entrance |
| `pulse-glow` | Box-shadow intensity oscillation | 3s infinite | Glow effects |
| `pulse` | Opacity oscillation | 2s infinite | Live dot |

---

## Streamlit Component Overrides

### Buttons
```css
.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    border: 1px solid var(--border-primary);
    background: var(--bg-card);
    transition: all 0.2s ease;
}
/* Hover: cyan border + glow */
/* Primary: brand gradient background */
```

### Tabs
```css
/* Tab bar: dark background, bordered */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border-primary);
}
/* Active tab: card bg, cyan text */
.stTabs [aria-selected="true"] {
    background: var(--bg-card);
    color: var(--accent-cyan);
}
```

### Data Tables
```css
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-primary);
    border-radius: 12px;
}
/* Header: dark bg, uppercase small text */
```

### Inputs
```css
/* Focus: cyan border ring */
.stSelectbox > div > div:focus-within {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.2);
}
```

---

## Chart Styling

All Plotly charts use:
```python
template="plotly_dark"
paper_bgcolor="rgba(0,0,0,0)"
plot_bgcolor="rgba(0,0,0,0)"
font=dict(family="Inter, sans-serif")
margin=dict(l=0, r=0, t=10, b=0)
```

### Chart Colors
| Element | Color |
|---------|-------|
| Candlestick (up) | `#10b981` (emerald) |
| Candlestick (down) | `#f43f5e` (rose) |
| SMA 20 | `#8b5cf6` (violet, dotted) |
| SMA 50 | `#f59e0b` (amber, dotted) |
| RSI | `#22d3ee` (cyan) |
| Volume | `rgba(34, 211, 238, 0.2)` |
| Grid lines | `rgba(51, 65, 85, 0.3)` |
| Efficient Frontier | `#22d3ee` (cyan) |
| Max Sharpe Star | `#8b5cf6` (violet) |
| Drawdown fill | `rgba(244, 63, 94, 0.1)` |

---

## Responsive Design

| Breakpoint | Behavior |
|-----------|----------|
| > 768px | Full 4-column grid, all metrics visible |
| ≤ 768px | 2-column grid, stacked layout |

---

## Hidden Defaults

The following Streamlit defaults are hidden via CSS:
```css
#MainMenu { visibility: hidden; }     /* Hamburger menu */
footer { visibility: hidden; }        /* "Made with Streamlit" */
header { visibility: hidden; }        /* Default header */
.stDeployButton { display: none; }    /* Deploy button */
```

---

## File Reference

All CSS is defined in `app.py` lines 29-415 (injected via `st.markdown` with `unsafe_allow_html=True`).

To modify the design system:
1. Edit the CSS variables in `:root` block (line 32-45) for global color changes
2. Edit component classes for specific component styling
3. Edit Streamlit overrides for Streamlit widget styling
4. All changes take effect on next page load (no rebuild needed)
