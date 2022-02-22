''' 
This file would output the specific data we need from each of the 
hotel_boxes in *BOOKING.PY* from the REPORT_RESULTS method.
'''
from typing import Collection
from selenium.webdriver.remote.webelement import WebElement

class BookingReport:
    def __init__(self, boxes_section_element:WebElement) -> None:
        self.boxes_section_element = boxes_section_element
        self.deal_boxes = self.pull_deal_boxes()
        
    def pull_deal_boxes(self):
        return self.boxes_section_element.find_elements_by_class_name(
            'sr_property_block'
        )

    def pull_deal_boxes_attributes(self):
        collection = []
        for deal_box in self.deal_boxes:
            # pulling the  HOTEL NAME
            hotel_name = deal_box.find_element_by_class_namae(
                'sr-hotel__name'
            ).get_attribute('innerHTML').strip()

            # pulling the HOTEL PRICE
            hotel_price = deal_box.find_element_by_class_name(
                'bui-price-display__value'
            ).get_attribute('innerHTML').strip()

            hotel_score = deal_box.get_attribute('data-score').strip()
            collection.append([hotel_name, hotel_price, hotel_score])
        return collection

