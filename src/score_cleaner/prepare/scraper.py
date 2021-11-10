import src.utils as utils

import bs4
import os
import pdf2image
import requests
import urllib

def download_scores():
    # TODO: handle case when images dir is already there
    """ Download the scores from the weimar jazz dataset.
    """
    db_content_url = 'https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html'
    page = requests.get(db_content_url)
    soup = bs4.BeautifulSoup(page.content, 'html.parser')
    links = soup.findAll(attrs = {'class': 'reference external'})
    score_elements = [e for e in links if e.get_text() == 'Score']
    score_paths = [e['href'] for e in score_elements]
    db_url_parsed = urllib.parse.urlparse(db_content_url)
    base_path = '/'.join(db_url_parsed.path.split('/')[:-1])
    score_paths = ['/'.join([base_path, p]) for p in score_paths]
    score_urls = []
    for score_path in score_paths:
        parsed_score = db_url_parsed
        parsed_score = parsed_score._replace(path=score_path)
        score_urls.append(parsed_score.geturl())
    configs = utils.Configs('score_cleaner')
    data_path = configs.data_path
    images_path = os.path.join(data_path, 'images')
    if not os.path.isdir(images_path):
        os.mkdir(images_path)
    for score_url in score_urls:
        response = requests.get(score_url)
        score_name = score_url.split('/')[-1][:-4]
        score_dir = os.path.join(images_path, score_name)
        os.mkdir(score_dir)
        try:
            pages = pdf2image.convert_from_bytes(response.content, 500)
        except pdf2image.exceptions.PDFPageCountError:
            log_path = ['src', 'score_cleaner', 'logs', 'download_errors.log']
            err_log = os.path.join(utils.PROJECT_ROOT, *log_path)
            with open(err_log, 'a+') as f:
                f.write(score_name)
            continue
        for i, page in enumerate(pages):
            file_path = os.path.join(score_dir, f'page_{i}.jpg')
            page.save(file_path, 'JPEG')
            
