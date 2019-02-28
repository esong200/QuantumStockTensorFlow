from newspaper import Article
url = "https://www.marketwatch.com/story/heres-a-better-buy-and-hold-strategy-using-the-dow-jones-industrial-average-2019-02-26"
a = Article(url, language = 'en') #English
a.download()
a.parse()
print(a.text)

