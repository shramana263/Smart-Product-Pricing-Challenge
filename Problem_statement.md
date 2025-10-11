
---

# 📦 Smart Product Pricing Challenge — ML Challenge 2025

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your task is to develop a machine learning solution that analyzes product details and predicts the price. The relationship between product attributes and pricing is complex — with factors like brand, specifications, and product quantity directly influencing the price.

Your objective is to build a model that can holistically analyze these product details and suggest an optimal price.

---

## 📂 Data Description

The dataset consists of the following columns:

1. **sample_id**
   Unique identifier for each input sample.

2. **catalog_content**
   Text field containing the product title, description, and Item Pack Quantity (IPQ), concatenated.

3. **image_link**
   Public URL to download the product image.
   Example: `https://m.media-amazon.com/images/I/71XfHPR36-L.jpg`
   Use the `download_images` function from `src/utils.py` (example in `src/test.ipynb`).

4. **price**
   Target variable (only available in training data).

---

## 📁 Dataset Details

* **Training dataset**: 75,000 products with complete product details and prices
* **Test dataset**: 75,000 products without price (to generate predictions)

---

## ✅ Output Format

Submit a CSV file with exactly **2 columns**:

| sample_id | price |
| --------- | ----- |

* `sample_id` must match the IDs in `test.csv`
* `price` must be a **positive float value**
* Number of rows must exactly match `test.csv`

---

## 📄 File Descriptions

### Source Files

* `src/utils.py`
  Contains helper functions to download images from `image_link`
  (Retry may be needed due to throttling)

* `sample_code.py`
  Optional dummy code to generate output in the required format

### Dataset Files

* `dataset/train.csv` — Training data with prices
* `dataset/test.csv` — Test data (without prices)
* `dataset/sample_test.csv` — Sample test input
* `dataset/sample_test_out.csv` — Sample output format (structure to follow)

---

## ⛔ Constraints

* Output **must match** the sample format exactly
* Predicted prices must be **positive floats**
* Final model must be MIT or Apache 2.0 licensed
* Model size must be **≤ 8 Billion parameters**

---

## 📊 Evaluation Metric — SMAPE

**Symmetric Mean Absolute Percentage Error (SMAPE)**:

[
\text{SMAPE} = \frac{1}{n} \sum \frac{|predicted_price - actual_price|}{\left(|actual_price| + |predicted_price|\right)/2}
]

**Example:**
Actual price = 100, Predicted price = 120
[
\text{SMAPE} = \frac{|100 - 120|}{(100 + 120)/2} \times 100% = 18.18%
]

* Range: **0% to 200%**
* Lower SMAPE = Better performance

---

## 🏆 Leaderboard

* **Public Leaderboard**: Based on 25K samples (real-time feedback)
* **Final Rankings**:

  * Based on the full 75K test set
  * Includes documentation review

---

## 📤 Submission Requirements

1. **Submit `test_out.csv`**

   * Must follow `sample_test_out.csv` format exactly

2. **Submit a 1-page document** covering:

   * Methodology
   * Selected model(s)
   * Feature engineering
   * Any relevant implementation details
     *(Use the template: `Documentation_template.md`)*

---

## 🚫 Academic Integrity — No External Price Lookup

Strictly prohibited:

* Web scraping product prices
* Using APIs to fetch market prices
* Manual lookups from websites
* Using external pricing databases

**Only the provided dataset may be used.**
Violations lead to **immediate disqualification**.

---

## ✅ Tips for Success

* Use both **textual features** (`catalog_content`) and **visual features** (images)
* Apply clever **feature engineering** on image + text
* Try **ensemble models** for better performance
* Handle **outliers** and preprocess data carefully

---

Let me know if you’d like this converted to PDF, DOCX, or split into structured sections for documentation!
