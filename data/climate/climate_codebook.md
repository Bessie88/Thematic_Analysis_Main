# Codebook: Climate Obstruction Narratives in Fossil Fuel Social Media Advertising

## Construct Definition

This construct refers to advertising narratives that present the fossil fuel industry in a positive, necessary, pragmatic, or socially beneficial way, thereby delaying, weakening, or redirecting climate action.

## Label Structure

Labels are organized into four super-categories, each containing one or more specific codes. A single ad may receive multiple labels (recorded in columns Typology 1–4).

---

### 1. Community & Resilience

| Code | Description |
|------|-------------|
| **CA** | The oil and gas sector contributes to local or national economies and communities, including through tax revenues, charitable efforts, and community support. |
| **CB** | The oil and gas sector creates jobs, sustains existing jobs, and supports workers' livelihoods. |

---

### 2. Green Innovation and Climate Solutions

| Code | Description |
|------|-------------|
| **GA** | The oil and gas sector is reducing emissions, setting climate targets, supporting climate policy, or investing in emissions-reduction technologies. |
| **GC** | "Clean" or "green" fossil fuels, especially natural gas or lower-carbon fuels, are presented as climate solutions. |

---

### 3. Pragmatism / Pragmatic Energy Mix

| Code | Description |
|------|-------------|
| **PA** | Oil and gas are presented as essential, reliable, affordable, safe, or pragmatic energy sources that are necessary for a functioning energy system. |
| **PB** | Oil and gas are presented as necessary raw materials for non-power uses and manufactured goods, such as plastics, medical supplies, clothing, or everyday products. |

---

### 4. Patriotic Energy Mix

| Code | Description |
|------|-------------|
| **SA** | Domestic oil and gas production is presented as beneficial to the nation, especially through energy independence, energy security, or energy leadership. |

---

## Data File

**`climate_real_data.csv`** — Each row is one social media ad. Columns:

| Column | Description |
|--------|-------------|
| `id` | Original ad identifier |
| `ad_creative_body` | Full text of the advertisement |
| `Typology 1` | Primary label (one of: CA, CB, GA, GC, PA, PB, SA, or blank if none apply) |
| `Typology 2–4` | Additional labels for ads with multiple narratives |

Blank label columns indicate the ad does not contain an obstruction narrative under this construct.
