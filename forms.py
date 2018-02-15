from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit

class selectClust(forms.Form):
    clust = forms.IntegerField(
        label='How much clusters? ',
        help_text='must between 3 and 6',
        min_value=3,
        max_value=6,
        required=True
    )

    def __init__(self, *args, **kwargs):
        super(selectClust, self).__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_class = 'form-horizontal'
        self.helper.label_class = 'col-lg-4'
        self.helper.field_class = 'col-lg-4'
        self.helper.form_method = 'POST'
        self.helper.form_action = ''

        self.helper.add_input(Submit('submit', 'Submit'))