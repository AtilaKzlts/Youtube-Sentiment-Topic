<div align="center">
  <h1>Customer Feedback Analysis on a New Product Launch</h1>
 </p>
</div>


## Table of Contents

+	Project Introduction
    +	Executive Summary
    +	About the Data Set
    +	Methodology
    +	Objective
    
+	Analysis Outputs
    +	Sentiment Analysis
    +	Trend Analysis
    +	Topic Modeling
    +	Question Detection 


## Project Introduction

This project presents a complete Natural Language Processing (NLP) pipeline developed to analyze customer feedback following a product launch. 

Using the official **YouTube Data API**, user comments were collected from product review videos and processed to uncover key insights about public perception.

The core objective was to help marketing and product teams better understand what users think about the product — what they like, what they are confused about, and what they frequently ask.

Techniques applied in this project include:
- **Sentiment Analysis** (BERT-based)
- **Topic Modeling** (BERTopic)
- **Question Detection**
- **Text Cleaning & Language Detection**
- **N-gram Extraction & Trend Visualization**

The final output delivers not only structured insights from unstructured user data, but also identifies high-interest discussion topics and key user questions.  
The approach is **product-agnostic** and can be reused for any type of product with online review content.

## Executive Summary 

* **Overall Sentiment:**
  54% of the comments are positive, while 46% are negative. The product is generally perceived favorably, but there are significant concerns.

* **Engagement Times:**
  Comments peak particularly between 7:00 PM and 9:00 PM, with positive feedback notably increasing around 6:00 PM.

* **Strengths:**
  The aesthetic design, performance, and the bright atmosphere of both interior and exterior spaces are frequently praised by customers. These features should be prominently highlighted in marketing efforts.

* **Areas for Improvement:**
  Critical areas requiring development include range capacity, technological modernity (e.g., references to “fax machine”), the crew cabin, and equipment/layout issues in living spaces.

* **High-Interest Topics:**
  White goods and auxiliary equipment, product naming, range and fuel efficiency, yacht equipment, and pricing are the most frequently asked-about subjects, necessitating detailed content and communication strategies.

* **Recommended Actions:**

  1. New product content and marketing campaigns should be scheduled according to peak engagement hours between 7:00 PM and 9:00 PM.
  2. Negative feedback (regarding range, technology, crew cabin, equipment) should be proactively addressed with transparent, detailed FAQs, blogs, and video content; collaboration with content creators for video production is recommended if necessary.
  3. Comprehensive, transparent, and clear information should be provided on technical specifications and pricing.
  4. The strong design and performance features should be emphasized in all promotional materials.

**Conclusion:**
These strategies will enhance customer satisfaction and strengthen the product’s market position. It is critical to prioritize and rapidly implement improvements in the identified areas.




## About the Data Set 

| Column Name    | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `video_id`     | Unique identifier of the YouTube video from which the comment was retrieved |
| `comment_id`   | Unique identifier of the comment                                            |
| `parent_id`    | ID of the parent comment if it's a reply; `None` for top-level comments     |
| `is_reply`     | Boolean indicating whether the comment is a reply or not (`True/False`)     |
| `published_at` | Timestamp when the comment was published                                    |
| `like_count`   | Number of likes the comment received                                        |
| `text`         | The original comment text as displayed on YouTube                           |


## Methodology

**Data Collection:**

* Retrieved comments using the YouTube API (specified endpoint, applied filters, date range, etc.)

**Data Cleaning:**

* Removed empty or irrelevant comments
* Performed language detection and filtered out comments not in Turkish or English
* Cleaned special characters and removed emojis
* Converted text to lowercase and standardized punctuation
* Removed URLs, email addresses, and user mentions

**Data Transformation:**

* Applied tokenization (word segmentation)
* Removed stop words (irrelevant/common words)
* Performed lemmatization or stemming
* Extracted n-grams for phrase-level analysis

**Sentiment Analysis Preparation:**

* Vectorized cleaned text using BERT embeddings
* Classified comments into positive, negative, or neutral sentiment categories

**Topic Modeling:**

* Utilized BERTopic to identify main discussion topics
* Reduced dimensionality using UMAP for visualization
* Clustered comments based on semantic similarity

**Question Detection:**

* Extracted questions using rule-based patterns and keyword matching
* Categorized questions by topic for focused content strategy

**Visualization and Reporting:**

* Created time-series charts for comment volume and sentiment trends
* Developed heatmaps for engagement and sentiment intensity by hour
* Mapped topic clusters and sentiment distribution per topic
* Generated actionable insights for marketing, product, and communication teams


----

## Objective 

* To understand the **public perception** of our new product (Omikron OT60 yacht) through YouTube comments.

* To determine the **overall sentiment** (positive, negative) in the comments and analyze if there is polarization.
* To identify the **main discussion topics** and **frequently asked questions** related to the product.
* To transform these comprehensive analysis results into actionable insights for guiding our **product development, marketing, and communication strategies**, and to **maximize user engagement**.

## Analysis Outputs

*Most comments are concise, averaging 92 characters, indicating quick user reactions.*

![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/1.png)

When the distribution is examined, it is observed that a significant portion of the comments are quite short, with the highest frequency concentrated between 0 and approximately 100 characters. In particular, comments within the 0–50 character range constitute the most crowded segment of the comment pool.

The red dashed line in the chart indicates that the average comment length is around 92 characters, which supports the notion that comments tend to be brief and to the point.

This suggests that users often express quick reactions, simple questions, or likes, while more in-depth or detailed opinions are relatively less frequent.

---
*Most comments are original and not replies, indicating users react more to content than to each other.*

![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/2.png)

The Reply Status Chart reveals that the vast majority of comments (marked as False, indicating top-level comments) are direct reactions to the video content itself, while user-to-user interactions (replies) remain relatively limited. This indicates that initial responses and personal opinions about the Omikron OT60 are more dominant than conversational engagement among users.

The Hourly Comment Count Chart demonstrates clear fluctuations in comment activity throughout the day. Notably, the peak period falls between 7:00 PM and 9:00 PM, when the number of comments reaches its highest levels.

This insight carries significant implications for the marketing and communication strategy of the Omikron OT60. Publishing new content or promotional material during this time window could potentially maximize user engagement and feedback collection.

---
*Sentiment analysis reveals a positive majority, with a significant negative minority.*

![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/3.png)

As observed, the comments are predominantly divided into two main sentiment categories: positive (910 comments) and negative (777 comments). Since the model does not assign a distinct label for neutral sentiment, the distribution reflects a general tendency toward positive perception, though a substantial portion of negative feedback is also present.

Approximately more than 5 out of every 10 comments are positive, while close to 4 are negative. This suggests that although the product overall leaves a favorable impression, there are still a considerable number of concerns or areas for improvement that warrant attention.

A deeper analysis of the negative comments can provide directly actionable insights for product development and customer support teams.

---
*Sentiment trends show a promotional event spiked both positive and negative reactions.*

![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/4.png)
![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/5.png)

When sentiment trends are examined separately for positive and negative comments, meaningful insights emerge regarding the product’s marketing and promotional impact.

The Positive Sentiment Chart shows a dramatic increase in October 2024, suggesting that a specific event or announcement during this period sparked strong excitement and favorable reactions among users. This spike likely reflects a successful promotional campaign or a significant positive development that was quickly embraced by the audience.

However, the Negative Sentiment Chart also reveals a noticeable rise in negative comments during the same high-engagement period. This indicates that heightened interest in the product brought not only praise but also criticism and unmet expectations.

Negative feedback emerging during peak attention periods may point to certain aspects of the product that failed to meet user expectations. These critical insights provide high-priority areas for review by the product development teams.

---
*Sentiment heatmap identifies peak hours of emotional engagement.*

![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/6.png)

This heatmap provides a detailed view of how sentiment intensity regarding the product varies throughout the day. The analysis reveals that user engagement and emotional expression peak during specific time windows.

In particular, the time frame between 5:00 PM and 8:00 PM stands out as the period with the highest overall comment activity, with noticeable increases in both positive and negative sentiments. The most prominent spike occurs at 6:00 PM, where 83 positive comments were recorded, indicating that the product receives the most favorable attention during this hour, and users are more likely to express their appreciation in that time slot.

Additionally, there is a significant level of activity around midnight (00:00), involving both positive and negative sentiments. In contrast, the early morning to mid-day period (2:00 AM – 1:00 PM) shows the lowest level of comment activity.

These insights serve as a valuable timing guide for optimizing content publication, social media campaigns, and audience engagement strategies related to the Omikron OT60. To maximize interaction with potential customers, the company could specifically target the evening peak hours in future outreach efforts.

---
*Topic map reveals key clusters of user discussion themes.*

![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/7.png)

The Topic Distance Map generated by BERTopic visually presents the semantic relationships and densities of the main discussion topics within the reviews about the Omikron OT60 mattress. The map shows that the comments are clustered around certain common themes, forming several distinct topic clusters. Notably, clusters concentrated in the upper left and upper right areas indicate that users have specific focal points in their discussions. The size of each circle represents the number of comments assigned to that topic, with larger circles indicating more frequently discussed topics. This visualization reveals the diversity of user opinions related to the Omikron OT60 and highlights which semantic areas are closely related, enabling the company to tailor its communication and product development strategies based on specific topic categories. The distances between topics serve as a guide to understand how different user segments or interest areas are differentiated.

---
*Sentiment distribution per topic highlights strengths and concerns.*

![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/8.png)

The Sentiment Distribution by Topic chart reveals the overall sentiment trends within the reviews about the Omikron OT60 mattress on a per-topic basis. While the majority of comments reflect a positive sentiment, certain topics show a significant proportion of negative feedback. This indicates that, although the product generally enjoys a favorable perception, there are specific areas where user dissatisfaction or opportunities for improvement exist. This analysis provides critical insights for the company to both reinforce its strengths and strategically address the topics with concentrated negative feedback.

---
*Topic-based sentiment analysis identifies detailed product pros and cons.*

![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/9.png)
![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/10.png)

The topic-based sentiment analysis of user reviews for the Omikron OT60 mattress clearly highlights both the product’s strengths and areas requiring improvement. Positive comments emphasize strong appreciation for the mattress’s aesthetic design, overall impressive performance, and the bright atmosphere it creates both indoors and outdoors. Conversely, negative feedback points to concerns and expectations for improvement, particularly regarding the mattress’s range capacity, technological modernity (such as references to outdated equipment like fax machines), the crew cabin, and certain specific equipment or layout issues within the general living areas.

---
*Question volume chart highlights topics generating the most viewer inquiries.*

![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/11.png)
![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/12.png)
![image](https://github.com/AtilaKzlts/Youtube-Sentiment-Topic/blob/main/assets/13.png)

This chart illustrates the distribution of viewer interest based on the volume of questions found in YouTube comments. According to the chart, "Topic 6" (Naming and Meaning Questions) and "Topic 21" (Appliances and Auxiliary Equipment Questions) contain the highest number of questions, each with 21 inquiries. This indicates that these topics are the most curious or uncertain areas among viewers. "Topic 3" (Yacht Equipment and Structural Questions), "Topic 4" (Range and Fuel Efficiency Questions), and "Topic 1" (General Yacht Ownership and Usage Questions) also contain a high volume of questions. Additionally, "Topic 16" (Pricing and Cost Questions) highlights the importance of pricing-related inquiries. In this context, content creators can enhance viewer engagement and better meet audience expectations by focusing future videos or communication strategies on frequently asked subjects such as naming conventions, yacht interior equipment (especially appliances), technical specifications (engine, range), and pricing, providing detailed explanations and Q&A-style content.
