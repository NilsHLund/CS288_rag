import json
import time
import re
import argparse
import os
import logging
import threading
from collections import deque
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm