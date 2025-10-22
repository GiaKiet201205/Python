from django.urls import path
from .views import calculate_routes
from .ops import switch_drivers_depot, add_customer, get_next_customer_id

urlpatterns = [
    path('calculate/', calculate_routes, name='calculate_routes'),
    path('switch-drivers/', switch_drivers_depot, name='switch_drivers_depot'),
    path('add-customer/', add_customer, name='add_customer'),
    path('next-customer-id/', get_next_customer_id, name='get_next_customer_id'),
]