''' 
This bot visits *BOOKING.COM*, changes the currency to USD, ask user for LOCATION INPUT, enters CHECK-IN
 & CHECK-OUT DATE, increases the number of Adult, clicks on the search button, filters the search results presented,
 creates a table with the (names, price & score) of the first 25 hotels presented. 
'''
from booking.booking import Booking
from booking.booking_report import BookingReport

try:
    with Booking() as bot:
        bot.land_first_page()
        bot.change_currency(currency='USD')
        bot.select_place_to_go(input('Enter location: '))
        print('Enter date in YYYY-MM-DD format.')
        bot.select_dates('2021-11-13', '2021-11-20')
        bot.select_adults(addition=2)
        bot.click_search()
        bot.apply_filtration()
        bot.refresh() # enables ur bot grab the hotel names properly
        bot.report_results()
        bot.close()
except Exception as e:
    if 'in PATH' in str(e):
        print(
            'You are trying to run the bot from command line \n'
            'Pease add to PATH your selenium Drivers \n'
            'Windows: \n'
            '   set PATH=%PATH%;C:path-to-your-folder \n \n'
            'Linux: \n'
            '   PATH=$PATH:/path/toyour/folder \n'
        )
    else:
        raise


