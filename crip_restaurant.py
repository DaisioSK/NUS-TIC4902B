from datetime import datetime
import requests
import time
import copy
import sys
import json

HEADERS = {
    "Content-Type": "application/json"
}

CSV_HEADERS = "username,rating,time,content"

FETCHED_SUCCESS = 200

CRAWLING_INTERVAL = 1

RESTAURANT_RESOURCE_URL = "https://m.ctrip.com/restapi/soa2/10332/GetHomePageRestaruantListV706?_fxpcqlniredt=09031032419531922213&x-traceID=09031032419531922213-1706149576963-3558926"

RESTAURANT_RESOURCE_PAYLOAD = {
    "PageIndex": 1,
    "PageSize": 49,
    "SubVersion": 1,
    "ViewDestId": 0,  # define the location of restaurant
    "head": {
        "cid": "09031032419531922213"
    }
}

RESTAURANT_REVIEW_RESOURCE_URL = "https://m.ctrip.com/restapi/soa2/13444/getCommentAndHotTagList?_fxpcqlniredt=09031032419531922213&x-traceID=09031032419531922213-1706152102604-2266126"

RESTAURANT_REVIEW_RESOURCE_MAX_PAGE_INDEX = 1

RESTAURANT_REVIEW_RESOURCE_PAYLOAD = {
    "arg": {
        "resourceId": "0", # define the restaurant
        "resourceType": 12,
        "pageIndex": 1,
        "pageSize": 49,
        "sortType": 3,
        "channelType": 5
    },
    "head": {
        "cid": "09031032419531922213"
    }
}


class RestaurantResource:
    def __init__(self, location, url, data, view_dest_id):
        self.location = location
        self.url = url
        self.data = copy.deepcopy(data)
        self.view_dest_id = view_dest_id
        # update the view_dest_id in data
        self.data["ViewDestId"] = view_dest_id


class Restaurant:
    def __init__(self, poi_id, restaurant_id, name, average_price, currency, score, landmark_name, landmark_distance, distance_desc, feature):
        self.poi_id = poi_id
        self.restaurant_id = restaurant_id
        self.name = name
        self.average_price = average_price
        self.currency = currency
        self.score = score
        self.landmark_name = landmark_name
        self.landmark_distance = landmark_distance
        self.distance_desc = distance_desc
        self.feature = feature
        self.review_list = []

        # set restaurant review request params
        # update resource_id in payload of review resource to respective restaurant_id
        self.review_resource_payload = copy.deepcopy(RESTAURANT_REVIEW_RESOURCE_PAYLOAD)
        self.review_resource_payload["arg"]["resourceId"] = str(restaurant_id)
        self.review_resource_max_page_index = RESTAURANT_REVIEW_RESOURCE_MAX_PAGE_INDEX  # fixed for now

    def set_review_list(self, review_list):
        self.review_list = review_list

    def get_csv_format(self):
        return "{}|{}|{}|{}|{}|{}|{}|{}|{}".format(self.restaurant_id, self.name, self.landmark_name, self.landmark_distance,
                                                self.distance_desc, self.average_price, self.currency, self.score, self.feature)

    def get_review_csv_format(self):
        content_list = list()
        for review in self.review_list:
            content_list.append(review.get_csv_format())
        return "\n".join(content_list)


class RestaurantReview:
    def __init__(self, restaurant_name, username, content, score, time):
        self.restaurant_name = restaurant_name
        self.username = username
        self.content = content
        self.score = score
        self.time = time

    def get_csv_format(self):
        return "{}|{}|{}|{}|{}".format(self.restaurant_name, self.username, self.score, self.time, self.content)


def get_restaurant_resource():
    singapore_restaurant_resource = RestaurantResource(
        location="Singapore",
        url=RESTAURANT_RESOURCE_URL,
        data=RESTAURANT_RESOURCE_PAYLOAD,
        view_dest_id=53)
    # add restaurant of other location here
    return [singapore_restaurant_resource]


def parse_response_to_restaurant_list(json_response):
    item_list = json_response["Restaurants"]
    if item_list is None or len(item_list) == 0:
        return None

    # store the parsed data
    restaurant_list = list()
    for item in item_list:
        poi_id = item.get("PoiId")
        restaurant_id = item.get("RestaurantId")
        name = item.get("Name")
        average_price = item.get("AveragePrice")
        currency = item.get("CurrencyUnit")
        score = item.get("CommentScore")
        landmark_name = item.get("LandmarkName")
        landmark_distance = item.get("LandmarkDistance")
        distance_desc = item.get("DistanceDesc")
        feature = item.get("Feature")
        restaurant = Restaurant(poi_id, restaurant_id, name, average_price, currency, score, landmark_name, landmark_distance, distance_desc, feature)
        restaurant_list.append(restaurant)

    return restaurant_list


def parse_response_to_review_list(restaurant_name, json_response):
    code = json_response["code"]
    if code != 200:
        return None

    result = json_response["result"]
    if result is None:
        return None

    item_list = result["commentInfoTypes"]
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
        time = parse_item_time(item["publishTime"])
        review = RestaurantReview(restaurant_name, username, content, score, time)
        review_list.append(review)
    return review_list


def parse_item_username(item):
    if item["userInfo"] is None:
        return False, ""
    return True, item["userInfo"]["userNick"]


def parse_item_content(content):
    # replace newline character (\n,\r) to the space (\s)
    return content.replace("\n", " ", -1).replace("\r", " ", -1)


def parse_item_time(publish_time):
    # process the data format first as the raw data is in the format of `/Date(1693614373000+0800)/`
    # we only need the timestamp part
    formatted_time = publish_time.replace("/", "", -1).replace("Date", "").replace("(", "").replace(")", "").replace(
        "+0800", "")
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


def persistent_data(location, restaurant_list):
    restaurant_csv_data_list = list()
    restaurant_review_csv_data_list = list()
    for restaurant in restaurant_list:
        restaurant_csv_data_list.append(restaurant.get_csv_format())
        restaurant_review_csv_data_list.append(restaurant.get_review_csv_format())

    # build content & write
    restaurant_csv_header = "id|name|landmark name|landmark distance|distance desc|average price|currency|score|feature"
    restaurant_review_csv_header = "restaurant name|username|score|review time|content"
    restaurant_csv_content = restaurant_csv_header + "\n" + "\n".join(restaurant_csv_data_list)
    restaurant_review_content = restaurant_review_csv_header + "\n" + "\n".join(restaurant_review_csv_data_list)
    restaurant_csv_file_name = write_as_csv("{}_{}".format(location, "restaurant"), restaurant_csv_content)
    restaurant_review_csv_file_name = write_as_csv("{}_{}".format(location, "restaurant_review"), restaurant_review_content)

    # show success message
    print("crawling for is done, the output filename is: `{}`, `{}`".format(location, restaurant_csv_file_name, restaurant_review_csv_file_name))


def write_as_csv(location, content):
    # csv filename follow the format of `location-timestamp`, i.e. xxxx-16000000.csv
    filename = "{}_{}.csv".format(location, int(datetime.now().timestamp()))
    with open(filename, "w") as f:
        f.write(content)
    return filename




def get_review_list_by_restaurant(restaurant):
    # list to collect all review data associated with the current restaurant
    review_list = list()
    review_resource_payload = restaurant.review_resource_payload

    # continue crawling until reach the max page size
    max_page_index = restaurant.review_resource_max_page_index
    current_page_index = review_resource_payload["arg"]["pageIndex"]
    while current_page_index <= max_page_index:
        # fetch remote data
        response = requests.post(url=RESTAURANT_REVIEW_RESOURCE_URL, headers=HEADERS, json=review_resource_payload)
        # check if the fetching is success
        if response.status_code == FETCHED_SUCCESS:
            # parse to response to python object
            review_list_of_current_page = parse_response_to_review_list(restaurant.name, response.json())
            # save the data of current page into review list
            review_list.extend(review_list_of_current_page)
        else:
            print("fail to get the resource from: {}, terminate the program ...".format(RESTAURANT_REVIEW_RESOURCE_URL))
            exit(0)

        # print success message
        print(
            "fetch review data success, restaurant name: {}, pageIndex: {}, wait for {} seconds before next crawling...".format(
                restaurant.name, current_page_index, CRAWLING_INTERVAL))

        # wait for 0.1 second before next crawling
        time.sleep(CRAWLING_INTERVAL)

        # move to next page and crawling
        current_page_index += 1
        review_resource_payload["arg"]["pageIndex"] = current_page_index

    # reviews associated with the resource should be collected in the form of python object when reach here
    return review_list


def start_crawling_for_restaurant_resource(restaurant_resource_list):
    for restaurant_resource in restaurant_resource_list:
        # store for consolidated data
        data_list = list()
        try:
            # build request param & send request
            response = requests.post(url=restaurant_resource.url, json=restaurant_resource.data)
            if response.status_code == FETCHED_SUCCESS:
                restaurant_list = parse_response_to_restaurant_list(response.json())
                if restaurant_list is None:
                    print("Restaurants attribute is not found from remote server, move to next restaurant resource")
                    continue
                # fetch the reviews from respective restaurant
                for restaurant in restaurant_list:
                    review_list = get_review_list_by_restaurant(restaurant)
                    restaurant.set_review_list(review_list)
                # save to data list
                data_list.extend(restaurant_list)
            else:
                print("fail to get the response from server, err: " + response.text)
                sys.exit(0)
        except Exception as exp:
            print("exception occur when send request to remote server, exception: ", exp)
            sys.exit(0)
        # write as csv file
        persistent_data(restaurant_resource.location, data_list)

def main():
    # prepare restaurant resource first
    restaurant_resource_list = get_restaurant_resource()
    # start
    start_crawling_for_restaurant_resource(restaurant_resource_list)


if __name__ == '__main__':
    main()
