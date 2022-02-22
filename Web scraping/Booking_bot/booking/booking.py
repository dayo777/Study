from selenium import webdriver
from selenium.webdriver.remote.webdriver import WebDriver
from booking.booking_report import BookingReport
import booking.constants as const
from booking.booking_filtration import BookingFiltration
from prettytable import PrettyTable

class Booking(webdriver.Firefox):
    def __init__(self, teardown=False):
        self.teardown = teardown
        # self.driver = driver_path
        options = webdriver.FirefoxOptions()
        # options.add_argument()
        super(Booking, self).__init__(options=options)
        self.implicitly_wait(15)
        self.maximize_window()

    def __exit__(self, *args) -> None:
        if self.teardown:
            return super().__exit__(*args)

    def land_first_page(self):
        self.get(const.BASE_URL)

    def change_currency(self, currency=None):
        currency_element = self.find_element_by_css_selector(
            'button[data-tooltip-text="Choose your currency"]'
        )
        currency_element.click()
        # note the ASTERICK before the equals sign
        selected_currency_element = self.find_element_by_css_selector(
            f'a[data-modal-header-async-url-param*="selected_currency={currency}"]'
        )
        selected_currency_element.click()

    def select_place_to_go(self, place_to_go):
        search_field = self.find_element_by_id('ss')
        search_field.clear()
        search_field.send_keys(place_to_go)
        first_result = self.find_element_by_css_selector(
            'li[data-i="0"]'
        )
        first_result.click()

    def select_dates(self, check_in, check_out):
        # date-format == YYYY-MM-DD
        check_in_date = self.find_element_by_css_selector(
            f'td[data-date="{check_in}"]'
        )
        check_in_date.click()

        check_out_date = self.find_element_by_css_selector(
            f'td[data-date="{check_out}"]'
        )
        check_out_date.click()

    def select_adults(self, subtract = None, addition = None):
        adult_selector = self.find_element_by_id("xp__guests__toggle")
        adult_selector.click()
        decrease_adult = self.find_element_by_css_selector(
            'button[aria-describedby="group_adults_desc"]'
        )
        increase_adult = self.find_element_by_css_selector(
            'button[aria-label="Increase number of Adults"]'
        )
        if subtract:
            for i in range(subtract):
                decrease_adult.click()

        if addition:
            for i in range(addition):
                increase_adult.click()

    def click_search(self):
        search_button = self.find_element_by_css_selector(
            'button[type="submit"]'
        )
        search_button.click()
        
    def apply_filtration(self):
        filtration = BookingFiltration()
        filtration.apply_star_rating(4,5)
        filtration.sort_price_lowest_first()

    def report_results(self):
        hotel_boxes = self.find_element_by_id(
            'hotellist_inner'
        )
        report = BookingReport(hotel_boxes)
        table = PrettyTable(field_names=['Hotel Name', 'Hotel Price', 'Hotel Score'])
        table.add_rows(report.pull_deal_boxes_attributes())
        print(table)
        
    def close(self) -> None:
        # return super().close()
        return webdriver.Firefox.quit()