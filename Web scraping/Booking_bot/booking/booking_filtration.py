'''
This file will include code to apply some filtrations after our SEARCH
has returned some results.
'''
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.remote.webdriver import WebDriver


class BookingFiltration:
    # intializing the type of the DRIVER object with WebDriver
    def __init__(self):
        self.driver = webdriver.Firefox()


    def apply_star_rating(self, *star_values):
        # element = WebDriverWait(self.driver, 10).until(
        #     EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-filters-group="class"]'))
        # )
        star_filtration_box = self.driver.find_element_by_css_selector(
            'div[data-filters-group="class"]'
        )
        star_child_elements = star_filtration_box.find_elements_by_class("_f38115ae0")
        for star_value in star_values:
            for star_element in star_child_elements:
                if str(star_element.get_attribute('innerHTML')).strip() == f'{star_value} stars':
                    star_element.click()
       
    def sort_price_lowest_first(self):
        element = self.driver.find_element_by_css_selector(
            'li[data-id="price"]'
        )
        element.click()

        



