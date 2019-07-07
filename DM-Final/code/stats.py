import pandas as pd
from nltk.tokenize import WordPunctTokenizer
word_tokenizer = WordPunctTokenizer()

def get_bigrams(unigrams):
    bigrams = []
    for i in range(len(unigrams) - 1):
        bigrams.append((unigrams[i], unigrams[i+1]))
    return bigrams

def analyze_tweet(tweet):
    result = dict()
    result['MENTIONS'] = tweet.count('userMentionToken')
    result['URLS'] = tweet.count('urlToken')
    result['POS_EMOS'] = tweet.count(':)')
    result['NEG_EMOS'] = tweet.count(':(')
    unigrams = word_tokenizer.tokenize(tweet)
    result['UNIGRAMS'] = len(unigrams)
    bigrams = get_bigrams(unigrams)
    result['BIGRAMS'] = len(bigrams)
    return result, unigrams, bigrams

def displayStatsInfo(file_path):
    
    processed_file_path = file_path
    all_data = pd.read_csv(processed_file_path)

    num_tweets, num_pos_tweets, num_neu_tweets, num_neg_tweets = 0, 0, 0, 0
    num_mentions, max_mentions = 0, 0
    num_emojis, num_pos_emojis, num_neg_emojis, max_emojis = 0, 0, 0, 0
    num_urls, max_urls = 0, 0
    num_unigrams, num_unique_unigrams, min_unigrams, max_unigrams = 0, 0, 1e6, 0
    num_bigrams, num_unique_bigrams = 0, 0

    all_unigrams = []
    all_bigrams = []
    
    num_tweets = len(all_data)
    
    for i in range(num_tweets):
        row = all_data.iloc[i]
        if row.sentiment == 'positive':
            num_pos_tweets += 1
        elif row.sentiment == 'neutral':
            num_neu_tweets += 1
        elif row.sentiment == 'negative':
            num_neg_tweets += 1
        result, unigrams, bigrams = analyze_tweet(row.text)
        num_mentions += result['MENTIONS']
        max_mentions = max(max_mentions, result['MENTIONS'])
        num_pos_emojis += result['POS_EMOS']
        num_neg_emojis += result['NEG_EMOS']
        max_emojis = max(max_emojis, result['POS_EMOS'] + result['NEG_EMOS'])
        num_urls += result['URLS']
        max_urls = max(max_urls, result['URLS'])
        num_unigrams += result['UNIGRAMS']
        min_unigrams = min(min_unigrams, result['UNIGRAMS'])
        max_unigrams = max(max_unigrams, result['UNIGRAMS'])
        all_unigrams.extend(unigrams)
        num_bigrams += result['BIGRAMS']
        all_bigrams.extend(bigrams)

    num_emojis = num_pos_emojis + num_neg_emojis
    unique_unigrams = list(set(all_unigrams))
    num_unique_unigrams = len(unique_unigrams)
    num_unique_bigrams = len(set(all_bigrams))
    
    print('\nAnalysis Statistics')
    print('Tweets => Total: {}, Positive: {}, Neutral: {}, Negative: {}'.format(num_tweets, 
                                                                                num_pos_tweets, 
                                                                                num_neu_tweets, 
                                                                                num_neg_tweets))
    print('User Mentions => Total: {}, Avg: {:.4f}, Max: {}'.format(num_mentions,
                                                                    num_mentions / float(num_tweets),
                                                                    max_mentions))
    print('URLs => Total: {}, Avg: {:.4f}, Max: {}'.format(num_urls, 
                                                           num_urls / float(num_tweets),
                                                           max_urls))
    print('Emojis => Total: {}, Positive: {}, Negative: {}, Avg: {:.4f}, Max: {}'.format(num_emojis,
                                                                                         num_pos_emojis,
                                                                                         num_neg_emojis,
                                                                                         num_emojis / float(num_tweets),
                                                                                         max_emojis))
    print('Unigrams => Total: {}, Unique: {}, Avg: {:.4f}, Max: {}, Min: {}'.format(num_tweets,
                                                                                    num_unique_unigrams,
                                                                                    num_unigrams / float(num_tweets),
                                                                                    max_unigrams,
                                                                                    min_unigrams))
    print('Bigrams => Total: {}, Unique: {}, Avg: {:.4f}'.format(num_bigrams,
                                                                 num_unique_bigrams,
                                                                 num_bigrams / float(num_tweets)))
    