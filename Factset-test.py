# Content API - FactSet Prices -Prices endpoint sample code snippet
# We can follow the same code snippet for remaining end points (dividends, splits, returns, shares) by changing the endpoint and input parameters.
# This snippet demonstrates basic features of the FactSet Prices API by walking through the following steps:
#        1. Import Python packages
#        2. Enter your Username and API Key for authorization
#        3. For each Prices API endpoint, create request objects and display the results in a Pandas DataFrame
#           a.Create a request object and set the parameters
#           b.Create a POST Request and Data Frame

# 1. Import the required packages
import requests
import json
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from pandas.io.json import json_normalize

# 2. Create a connection object
# Enter your credentials for 'Username' and 'API Key' variables below.
# To generate an API key, visit (https://developer.factset.com/authentication) for more details on Authentication.

authorization = ('UNSW_AUS-978970','3p4hs28Ow2XKqbdbiz9PRerHKaTKocTwEOOluLIS')

fundamentals_endpoint = 'https://api.factset.com/content/factset-fundamentals/v1/fundamentals'
fundamentals_request={
  "ids": [
    "FDS-US"
  ],
  "periodicity": "QTR",
  "fiscalPeriodStart": "2017-09-01",
  "fiscalPeriodEnd": "2019-03-01",
  "metrics": [
    "FF_PAY_ACCT"
  ],
  "currency": "USD",
  "restated": "RP"
}

headers = {'Accept': 'application/json','Content-Type': 'application/json'}

# 3.2b `/factset-fundamentals/v1/fundamentals` - Pull data, display datafame properties, show initial records
# Create a POST Request
fundamentals_post = json.dumps(fundamentals_request)
fundamentals_response = requests.post(url = fundamentals_endpoint, data=fundamentals_post, auth = authorization, headers = headers, verify= False )
print('HTTP Status: {}'.format(fundamentals_response.status_code))
#create a dataframe from POST request, show dataframe properties
fundamentals_data = json.loads(fundamentals_response.text)
fundamentals_df = json_normalize(fundamentals_data['data'])
print('COLUMNS:')
print('')
print(fundamentals_df.dtypes)
print('')
print('RECORDS:',len(fundamentals_df))
# Display the Records
print(fundamentals_df[['fsymId','requestId','metric','currency','reportDate','periodicity','epsReportDate','fiscalEndDate','fiscalPeriod','fiscalPeriodLength','fiscalYear','updateStatus','updateType','value']].tail())