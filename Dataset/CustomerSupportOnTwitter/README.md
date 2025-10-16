Customer Support on Twitter (TWCS) dataset
=========================================

This folder stores a sample CSV excerpt from the Kaggle "Customer Support on Twitter" dataset (TWCS). The full corpus contains ~3 million tweets across 2.9M user-brand interactions. Kaggle requires an authenticated account to download the full dataset: https://www.kaggle.com/datasets/thoughtvector/customer-support-tweets

Files
-----
- sample.csv — 1,000-example excerpt published publicly on GitHub (naman-tiwari/Customer-Support-on-Twitter) for quick inspection.

How to obtain the complete data
-------------------------------
1. Create/sign in to a Kaggle account.
2. Accept the dataset terms on the Kaggle page above.
3. Configure your Kaggle API credentials (either create `~/.kaggle/kaggle.json` with `username` and `key`, or export the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).
4. Use the Kaggle CLI (already installed in this environment) to run:

   kaggle datasets download -d thoughtvector/customer-support-tweets -p Dataset/CustomerSupportOnTwitter
   unzip customer-support-tweets.zip -d Dataset/CustomerSupportOnTwitter

This will populate the folder with `twcs.csv` containing the full tweet threads.
