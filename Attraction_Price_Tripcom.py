import copy
import time
import csv
import requests
from datetime import datetime

# Replace 'your_api_endpoint' with the actual API endpoint URL
headers = {
    "Content-Type": "application/json;charset=UTF-8"
}

CRAWLING_INTERVAL = 0.1
URL = "https://sg.trip.com/restapi/soa2/21052/json/getProductShelf?_fxpcqlniredt=09031128319416176642&x-traceID=09031128319416176642-1705927370464-7945446"
DATA = {
  "clientInfo": {
    "currency": "SGD",
    "locale": "en-SG",
    "pageId": "10650006154",
    "channelId": 116,
    "platformId": 24,
  },
  "poiId": 89732,
}

def get_resource_list():

    return [
        Resource("USS_sales", 89732),
        Resource("Garden_sales",  94795),
        Resource("Safari_sales", 10758318),
        Resource("Sentosa_sales", 82520),
        Resource("Marina Bay Sands_sales", 10354609),
        Resource("Skypark_sales",  15131377),
        Resource("SEA Aquarium_sales", 101913),
        Resource("River Wonder_sales", 98700),
        Resource("Singapore Zoo_sales",  76136),
        Resource("Singapore Flyer_sales",  10758467),
        Resource("Botanic Garden_sales",  76133),
    ]


class Resource:
    def __init__(self, name, poiid):
        request_data = copy.deepcopy(DATA)
        request_data["poiId"] = poiid
        self.data = request_data
        self.name = name


def write_to_csv(data_list, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for line in data_list:
            writer.writerow(line.split('|'))


def crawl_resource(resource_list):

    for resource in resource_list:
        csv_data_list = ["ticket_name|price|currency|ticket_count|revenue|more_info"]
        data = resource.data
        response = requests.post(URL, json=data, headers=headers)
        if response.status_code == 200:
            price_resource_list = response.json()["resources"]
            for price_resource in price_resource_list:
                price = price_resource["priceInfo"]["price"]
                currency = price_resource["priceInfo"]["currency"]
                ticket_count = price_resource["statisticInfo"]["saleCount"]
                revenue = int(ticket_count) * int(price)
                ticket_name = price_resource["minPriceRelationInfo"]["peoplePropertyName"]
                ticket_info = "-" if price_resource.get("name") is None else price_resource.get("name")
                csv_data = "{}|{}|{}|{}|{}|{}".format(ticket_name,price,currency,ticket_count,revenue,ticket_info)
                csv_data_list.append(csv_data)
            time.sleep(CRAWLING_INTERVAL)
            print(
            "fetch data success,name: {}. wait for {} seconds before next crawling...".format(
                resource.name, CRAWLING_INTERVAL))
        else:
            print("Error:", response.status_code, response.text)
        filename = "{}.csv".format(resource.name)
        write_to_csv(csv_data_list, filename)


def main():
    resource_list = get_resource_list()
    crawl_resource(resource_list)

if __name__ == '__main__':
    main()

