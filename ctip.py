from datetime import datetime
import requests
import csv
import time
import copy

HEADERS = {
    "Content-Type": "application/json"
}

CSV_HEADERS = "username|rating|content|ip_address|time"

FETCHED_SUCCESS = 200

CRAWLING_INTERVAL = 1

URL = "https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList"
DATA = dict(arg=dict(channelType=2, collapseType=0, commentTagId=0, pageIndex=1, pageSize=50, poiId=82520, sourceType=1,
                     sortType=3, starType=0),
            head={"cid": "09031177116206593320", "ctok": "", "cver": "1.0", "lang": "01", "sid": "8888",
                  "syscode": "09", "auth": "", "xsid": "", "extension": []})
class Resource:
    def __init__(self, location, max_page_index, poiID ):
        request_data = copy.deepcopy(DATA)
        request_data["arg"]["poiId"] = poiID
        self.location = location
        self.max_page_index = max_page_index
        self.data = request_data



class Review:
    def __init__(self, username, content, score, ip, time):
        self.username = username
        self.content = content
        self.score = score
        self.ip = ip
        self.time = time

    def get_csv_format(self):
        return "{}|{}|{}|{}|{}".format(self.username, self.score, self.content,self.ip, self.time)


def get_resource_list():
    resource_list = [
        Resource("Sentosa", 50, 82520),
        Resource("USS", 50, 89732),
        Resource("Safari", 50, 10758318),
        Resource("Garden by the bay", 50, 94795),
        Resource("Marina Bay Sands", 10, 10354609),
        Resource("Skypark", 18, 15131377),
        Resource("SEA Aquarium", 50, 101913),
        Resource("River Wonder", 30, 98700),
        Resource("Singapore Zoo", 50, 76136),
        Resource("Singapore Flyer", 55, 10758467),
        Resource("Botanic Garden", 12, 76133),
    ]
    return resource_list


def crawling_and_save_as_csv(resource_list):
    for resource in resource_list:
        data = resource.data
        location = resource.location

        # list to collect all review data associated with the current resource
        review_list = list()

        # continue crawling until reach the max page size
        max_page_index = resource.max_page_index
        current_page_index = data["arg"]["pageIndex"]
        while current_page_index <= max_page_index:
            # fetch remote data
            response = requests.post(url=URL, headers=HEADERS, json=data)
            # check if the fetching is success
            if response.status_code == FETCHED_SUCCESS:
                # parse to response to python object
                review_list_of_current_page = parse_response_to_review_list(response.json())
                # save the data of current page into review list
                review_list.extend(review_list_of_current_page)
            else:
                print("fail to get the resource from: {}, terminate the program ...".format(URL))
                exit(0)

            # print success message
            print("fetch data success, resource location: {}, pageIndex: {}, wait for {} seconds before next crawling...".format(location, current_page_index, CRAWLING_INTERVAL))

            # wait for 0.1 second before next crawling
            time.sleep(CRAWLING_INTERVAL)

            # move to next page and crawling
            current_page_index += 1
            data["arg"]["pageIndex"] = current_page_index

        # reviews associated with the resource should be collected in the form of python object when reach here
        # write python object to csv content
        write_as_csv_content(location, review_list)


def write_as_csv_content(location, review_list):
    # convert the list to the csv content
    csv_data_list = list()
    csv_data_list.append(CSV_HEADERS)
    for review in review_list:
        csv_data_list.append(review.get_csv_format())

    # csv filename follow the format of `location-timestamp`, i.e. Sentosa-16000000.csv
    filename = "{}-{}.csv".format(location, int(datetime.now().timestamp()))
    with open(filename, "w", newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        for csv_data in csv_data_list:
            csv_writer.writerow(csv_data.split('|'))

    # show success message
    print("crawling for location: `{}` is done, the output filename is: `{}`".format(location, filename))


def parse_response_to_review_list(json_response):
    code = json_response["code"]
    if code != 200:
        return None

    result = json_response["result"]
    if result is None:
        return None

    item_list = result["items"]
    if len(item_list) == 0:
        return None

    review_list = list()
    for item in item_list:
        # username may not be present, if is absent, skip the current review data
        ok, username = parse_item_username(item)
        if not ok:
            continue
        content = parse_item_content(item["content"])
        score = parse_item_score(item["scores"])
        ip_address = parse_item_ip_address(item["ipLocatedName"])
        time = parse_item_time(item["publishTime"])
        review = Review(username, content, score, ip_address, time)
        review_list.append(review)
    return review_list


def parse_item_username(item):
    if item["userInfo"] is None:
        return False, ""
    return True, item["userInfo"]["userNick"]

def parse_item_content(content):
    # replace newline character (\n,\r) to the space (\s)
    content = content.replace(" | ", "-")
    content = content.replace("|", "-")
    return content.replace("\n", " ", -1).replace("\r", " ", -1)


def parse_item_time(publish_time):
    # process the data format first as the raw data is in the format of `/Date(1693614373000+0800)/`
    # we only need the timestamp part
    formatted_time = publish_time.replace("/", "", -1).replace("Date", "").replace("(", "").replace(")", "").replace("+0800", "")
    timestamp_in_milisec = int(formatted_time) / 1000
    return datetime.utcfromtimestamp(timestamp_in_milisec).strftime("%d/%m/%Y")


def parse_item_score(score_list):
    if len(score_list) == 0:
        return "0.0"
    # score is computed based on the avg of the total score in the list
    total = 0
    for score in score_list:
        total += score["score"]
    return "{:.1f}".format(total / len(score_list))

def parse_item_ip_address(item):
    if isinstance(item, dict) and "ipLocatedName" in item:
        return item["ipLocatedName"]
    elif isinstance(item, str):
        return item  # If item is a string, return it as is
    else:
        return "N/A"  # Default value if IP address is not available
def main():
    resource_list = get_resource_list()
    crawling_and_save_as_csv(resource_list)


if __name__ == '__main__':
    main()
