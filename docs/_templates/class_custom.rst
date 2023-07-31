{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
    :members:
    :show-inheritance:
    :inherited-members:
    :undoc-members:
    :member-order: groupwise

    .. automethod:: __init__

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
    {% for item in attributes %}
      ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block methods %}
    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
    {% for item in methods %}
      ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block dia %}
    .. rubric:: {{ _('Inheritance Diagram') }}

    .. inheritance-diagram:: {{ fullname }}
    {% endblock %}

