import logging
from dataclasses import dataclass
from typing import List, Set, Callable, Optional

import regex
from bohrapi.core import Artifact

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class Identity(Artifact):
    """
    >>> Identity({}).normalized_email is None
    True
    """
    @property
    def id(self) -> Optional[str]:
        if 'ids' not in self.raw_data:
            return None
        ids = self.raw_data['ids']
        if len(ids) > 1:
            raise ValueError(f'Multiple ids are available: {ids}')
        return ids[0] if ids[0] != '' else None

    @property
    def ids(self) -> Set[str]:
        return set(self.raw_data['ids']) if 'id' in self.raw_data else set()

    @property
    def email(self) -> Optional[str]:
        if 'emails' not in self.raw_data:
            return None
        emails = self.raw_data['emails']
        if len(emails) > 1:
            raise ValueError(f'Multiple emails are available: {emails}')
        return emails[0] if emails[0] != '' else None

    @property
    def emails(self) -> Set[str]:
        return set(self.raw_data['emails']) if 'emails' in self.raw_data else set()

    @property
    def name(self) -> Optional[str]:
        if 'names' not in self.raw_data:
            return None
        names = self.raw_data['names']
        if len(names) > 1:
            raise ValueError(f'Multiple names are available: {names}')
        return names[0] if names[0] != '' else None

    @property
    def normalized_name(self) -> Optional[str]:
        name = self.name
        if name is None:
            return None
        return apply_transforms(self.name, [
            remove_jr,
            remove_comma_between_names,
            remove_common_titles,
            remove_punctiation,
            normalize_spaces,
            lambda s: s.trim(),
        ])

    @property
    def normalized_email(self) -> Optional[str]:
        email = self.email
        if email is None:
            return None
        return apply_transforms(email, [
            remove_email_domain,
            remove_common_titles,
            remove_punctiation,
            lambda s: s.strip(),
        ])

    @property
    def names(self) -> Set[str]:
        return set(self.raw_data['names']) if 'names' in self.raw_data else set()

    def __repr__(self):
        return str(self)

    @staticmethod
    def important_fields_map():
        return {'email': 'email'}

    def __str__(self):
        return str(self.raw_data)


jr_re = regex.compile("[, ]*jr[. ]")
common_re = regex.compile("(mailing)?lists?|admin(istrator)?|webmaster|apache|support|postmaster|(un)?subscribe|dev|root|info|httpd|archive|mail(er)?|daemon|paypal|the|postgres?(ql)?|pgsql|hackers|lists?|python|dev|mysql")
punctuation_re = regex.compile("[,.-]*")
space_re = regex.compile(" [ ]+")


def remove_email_domain(email: str) -> str:
    """
    >>> remove_email_domain("hlibbabii@gmail.com")
    'hlibbabii'
    """
    return regex.sub("@.*", "", email)


def normalize_spaces(name: str) -> str:
    """
    >>> normalize_spaces(" Hlib     Babii  ")
    ' Hlib Babii '
    """
    return regex.sub(space_re, " ", name)


def remove_common_titles(name: str) -> str:
    """
    >>> remove_common_titles("python dev Hlib Babii")
    '  Hlib Babii'
    """
    return regex.sub(common_re, "", name)


def remove_punctiation(name: str) -> str:
    """
    >>> remove_punctiation("Hlib,, the admin.")
    'Hlib the admin
    """
    return regex.sub(punctuation_re, "", name)


def remove_jr(name: str) -> str:
    """
    >>> remove_jr("Bronny James jr.")
    'Bronny James'
    >>> remove_jr("Bronny James, jr.")
    'Bronny James'
    """
    return regex.sub(jr_re, "", name)


def remove_comma_between_names(name: str) -> 'str':
    """
    >>> remove_comma_between_names("John Smith")
    'John Smith'
    >>> remove_comma_between_names("Smith, John, the programmer")
    'Smith, John, the programmer'
    >>> remove_comma_between_names("Smith, John")
    ' John Smith'
    >>> remove_comma_between_names("Smith , John")
    ' John Smith '
    >>> remove_comma_between_names("John Smith, the programmer")
    'John Smith, the programmer'
    """
    spl = name.split(',')
    if len(spl) != 2 or len(spl[0].strip(" ").split(" ")) != 1:
        return name
    last_name, first_name = spl
    return f'{first_name} {last_name}'


def apply_transforms(name: str, transforms: List[Callable]) -> str:
    for transform in transforms:
        name = transform(name)
    return name
