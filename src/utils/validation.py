"""
Input Validation and Security Framework
Provides comprehensive input validation, sanitization, and security checks.
"""
import re
import html
import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from enum import Enum
import logging

from src.utils.error_handling import ValidationError, ErrorSeverity
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

class ValidationType(Enum):
    """Types of validation available"""
    STRING = "string"
    EMAIL = "email"
    NUMERIC = "numeric"
    DATE = "date"
    SYMBOL = "symbol"
    PRICE = "price"
    QUANTITY = "quantity"
    PERCENTAGE = "percentage"
    API_KEY = "api_key"
    SQL_SAFE = "sql_safe"

@dataclass
class ValidationRule:
    """Validation rule configuration"""
    validation_type: ValidationType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    error_message: Optional[str] = None

class InputValidator:
    """Comprehensive input validation system"""
    
    def __init__(self):
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'validation_by_type': {}
        }
        
        # Common regex patterns
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'symbol': r'^[A-Z]{1,5}$',  # Stock symbols
            'api_key': r'^[A-Za-z0-9]{16,}$',  # Minimum 16 character alphanumeric
            'safe_string': r'^[a-zA-Z0-9\s\-_.()]+$',  # Safe characters only
            'sql_injection': r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b|[\'";])',
            'xss': r'(<script|javascript:|on\w+\s*=)'
        }
    
    def validate_field(self, value: Any, rule: ValidationRule, field_name: str = "field") -> Any:
        """
        Validate a single field according to its rule
        
        Returns:
            Validated and sanitized value
            
        Raises:
            ValidationError: If validation fails
        """
        self.validation_stats['total_validations'] += 1
        validation_type = rule.validation_type.value
        self.validation_stats['validation_by_type'][validation_type] = \
            self.validation_stats['validation_by_type'].get(validation_type, 0) + 1
        
        try:
            # Check if required
            if rule.required and (value is None or value == ""):
                raise ValidationError(
                    f"Field '{field_name}' is required",
                    severity=ErrorSeverity.MEDIUM,
                    details={'field': field_name, 'rule': rule.validation_type.value}
                )
            
            # If not required and empty, return None
            if not rule.required and (value is None or value == ""):
                return None
            
            # Type-specific validation
            validated_value = self._validate_by_type(value, rule, field_name)
            
            # Length validation
            if hasattr(validated_value, '__len__'):
                self._validate_length(validated_value, rule, field_name)
            
            # Range validation
            if isinstance(validated_value, (int, float, Decimal)):
                self._validate_range(validated_value, rule, field_name)
            
            # Pattern validation
            if rule.pattern and isinstance(validated_value, str):
                if not re.match(rule.pattern, validated_value):
                    raise ValidationError(
                        f"Field '{field_name}' does not match required pattern",
                        severity=ErrorSeverity.MEDIUM,
                        details={'field': field_name, 'pattern': rule.pattern, 'value': str(validated_value)[:50]}
                    )
            
            # Allowed values validation
            if rule.allowed_values and validated_value not in rule.allowed_values:
                raise ValidationError(
                    f"Field '{field_name}' must be one of: {', '.join(map(str, rule.allowed_values))}",
                    severity=ErrorSeverity.MEDIUM,
                    details={'field': field_name, 'allowed_values': rule.allowed_values}
                )
            
            # Custom validation
            if rule.custom_validator and not rule.custom_validator(validated_value):
                error_msg = rule.error_message or f"Custom validation failed for field '{field_name}'"
                raise ValidationError(
                    error_msg,
                    severity=ErrorSeverity.MEDIUM,
                    details={'field': field_name}
                )
            
            return validated_value
            
        except ValidationError:
            self.validation_stats['failed_validations'] += 1
            raise
        except Exception as e:
            self.validation_stats['failed_validations'] += 1
            raise ValidationError(
                f"Validation error for field '{field_name}': {str(e)}",
                severity=ErrorSeverity.HIGH,
                details={'field': field_name, 'original_error': str(e)},
                original_error=e
            )
    
    def _validate_by_type(self, value: Any, rule: ValidationRule, field_name: str) -> Any:
        """Validate value based on its type"""
        validation_type = rule.validation_type
        
        if validation_type == ValidationType.STRING:
            return self._validate_string(value, field_name)
        elif validation_type == ValidationType.EMAIL:
            return self._validate_email(value, field_name)
        elif validation_type == ValidationType.NUMERIC:
            return self._validate_numeric(value, field_name)
        elif validation_type == ValidationType.DATE:
            return self._validate_date(value, field_name)
        elif validation_type == ValidationType.SYMBOL:
            return self._validate_symbol(value, field_name)
        elif validation_type == ValidationType.PRICE:
            return self._validate_price(value, field_name)
        elif validation_type == ValidationType.QUANTITY:
            return self._validate_quantity(value, field_name)
        elif validation_type == ValidationType.PERCENTAGE:
            return self._validate_percentage(value, field_name)
        elif validation_type == ValidationType.API_KEY:
            return self._validate_api_key(value, field_name)
        elif validation_type == ValidationType.SQL_SAFE:
            return self._validate_sql_safe(value, field_name)
        else:
            return value
    
    def _validate_string(self, value: Any, field_name: str) -> str:
        """Validate and sanitize string input"""
        if not isinstance(value, str):
            value = str(value)
        
        # Check for potential XSS
        if re.search(self.patterns['xss'], value, re.IGNORECASE):
            logger.warning(f"Potential XSS attempt in field '{field_name}'")
            raise ValidationError(
                f"Field '{field_name}' contains potentially harmful content",
                severity=ErrorSeverity.HIGH,
                details={'field': field_name, 'xss_detected': True}
            )
        
        # HTML escape for safety
        sanitized = html.escape(value.strip())
        return sanitized
    
    def _validate_email(self, value: Any, field_name: str) -> str:
        """Validate email format"""
        email_str = self._validate_string(value, field_name).lower()
        
        if not re.match(self.patterns['email'], email_str):
            raise ValidationError(
                f"Field '{field_name}' must be a valid email address",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name}
            )
        
        return email_str
    
    def _validate_numeric(self, value: Any, field_name: str) -> Union[int, float]:
        """Validate numeric input"""
        try:
            if isinstance(value, str):
                # Remove common formatting
                clean_value = value.replace(',', '').replace('$', '').strip()
                if '.' in clean_value:
                    return float(clean_value)
                else:
                    return int(clean_value)
            elif isinstance(value, (int, float)):
                return value
            else:
                raise ValueError(f"Cannot convert {type(value)} to numeric")
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"Field '{field_name}' must be a valid number",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'value': str(value)[:50]},
                original_error=e
            )
    
    def _validate_date(self, value: Any, field_name: str) -> datetime.date:
        """Validate date input"""
        if isinstance(value, datetime.date):
            return value
        elif isinstance(value, datetime.datetime):
            return value.date()
        elif isinstance(value, str):
            try:
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d']:
                    try:
                        return datetime.datetime.strptime(value, fmt).date()
                    except ValueError:
                        continue
                raise ValueError("No matching date format found")
            except ValueError as e:
                raise ValidationError(
                    f"Field '{field_name}' must be a valid date (YYYY-MM-DD)",
                    severity=ErrorSeverity.MEDIUM,
                    details={'field': field_name, 'value': value},
                    original_error=e
                )
        else:
            raise ValidationError(
                f"Field '{field_name}' must be a date",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'type': type(value).__name__}
            )
    
    def _validate_symbol(self, value: Any, field_name: str) -> str:
        """Validate stock symbol"""
        symbol = self._validate_string(value, field_name).upper()
        
        if not re.match(self.patterns['symbol'], symbol):
            raise ValidationError(
                f"Field '{field_name}' must be a valid stock symbol (1-5 uppercase letters)",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'value': symbol}
            )
        
        return symbol
    
    def _validate_price(self, value: Any, field_name: str) -> Decimal:
        """Validate price input with proper decimal handling"""
        try:
            if isinstance(value, Decimal):
                price = value
            else:
                numeric_value = self._validate_numeric(value, field_name)
                price = Decimal(str(numeric_value))
            
            if price < 0:
                raise ValidationError(
                    f"Field '{field_name}' must be a positive price",
                    severity=ErrorSeverity.MEDIUM,
                    details={'field': field_name, 'value': str(price)}
                )
            
            if price > 1000000:  # $1M max price sanity check
                raise ValidationError(
                    f"Field '{field_name}' exceeds maximum allowed price",
                    severity=ErrorSeverity.MEDIUM,
                    details={'field': field_name, 'value': str(price)}
                )
            
            return price.quantize(Decimal('0.01'))  # Round to cents
            
        except (InvalidOperation, TypeError) as e:
            raise ValidationError(
                f"Field '{field_name}' must be a valid price",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'value': str(value)[:50]},
                original_error=e
            )
    
    def _validate_quantity(self, value: Any, field_name: str) -> int:
        """Validate share quantity"""
        quantity = self._validate_numeric(value, field_name)
        
        if not isinstance(quantity, int) and not quantity.is_integer():
            raise ValidationError(
                f"Field '{field_name}' must be a whole number of shares",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'value': quantity}
            )
        
        quantity = int(quantity)
        
        if quantity < 0:
            raise ValidationError(
                f"Field '{field_name}' must be a positive quantity",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'value': quantity}
            )
        
        if quantity > 1000000:  # 1M shares max sanity check
            raise ValidationError(
                f"Field '{field_name}' exceeds maximum allowed quantity",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'value': quantity}
            )
        
        return quantity
    
    def _validate_percentage(self, value: Any, field_name: str) -> float:
        """Validate percentage (0-100)"""
        percentage = self._validate_numeric(value, field_name)
        
        if percentage < 0 or percentage > 100:
            raise ValidationError(
                f"Field '{field_name}' must be between 0 and 100",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'value': percentage}
            )
        
        return float(percentage)
    
    def _validate_api_key(self, value: Any, field_name: str) -> str:
        """Validate API key format"""
        api_key = self._validate_string(value, field_name)
        
        if not re.match(self.patterns['api_key'], api_key):
            raise ValidationError(
                f"Field '{field_name}' must be a valid API key (minimum 16 alphanumeric characters)",
                severity=ErrorSeverity.HIGH,
                details={'field': field_name}
            )
        
        return api_key
    
    def _validate_sql_safe(self, value: Any, field_name: str) -> str:
        """Validate string is safe from SQL injection"""
        safe_string = self._validate_string(value, field_name)
        
        # Check for SQL injection patterns
        if re.search(self.patterns['sql_injection'], safe_string, re.IGNORECASE):
            logger.error(f"Potential SQL injection attempt in field '{field_name}'")
            raise ValidationError(
                f"Field '{field_name}' contains potentially harmful SQL content",
                severity=ErrorSeverity.CRITICAL,
                details={'field': field_name, 'sql_injection_detected': True}
            )
        
        return safe_string
    
    def _validate_length(self, value: Any, rule: ValidationRule, field_name: str):
        """Validate length constraints"""
        length = len(value)
        
        if rule.min_length and length < rule.min_length:
            raise ValidationError(
                f"Field '{field_name}' must be at least {rule.min_length} characters long",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'min_length': rule.min_length, 'actual_length': length}
            )
        
        if rule.max_length and length > rule.max_length:
            raise ValidationError(
                f"Field '{field_name}' must be no more than {rule.max_length} characters long",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'max_length': rule.max_length, 'actual_length': length}
            )
    
    def _validate_range(self, value: Union[int, float, Decimal], rule: ValidationRule, field_name: str):
        """Validate numeric range constraints"""
        if rule.min_value is not None and value < rule.min_value:
            raise ValidationError(
                f"Field '{field_name}' must be at least {rule.min_value}",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'min_value': rule.min_value, 'actual_value': value}
            )
        
        if rule.max_value is not None and value > rule.max_value:
            raise ValidationError(
                f"Field '{field_name}' must be no more than {rule.max_value}",
                severity=ErrorSeverity.MEDIUM,
                details={'field': field_name, 'max_value': rule.max_value, 'actual_value': value}
            )
    
    def validate_dict(self, data: Dict[str, Any], rules: Dict[str, ValidationRule]) -> Dict[str, Any]:
        """
        Validate a dictionary of values according to rules
        
        Returns:
            Dict with validated and sanitized values
        """
        validated_data = {}
        errors = []
        
        # Validate all fields
        for field_name, rule in rules.items():
            try:
                value = data.get(field_name)
                validated_data[field_name] = self.validate_field(value, rule, field_name)
            except ValidationError as e:
                errors.append(e)
        
        # Check for unexpected fields in strict mode
        unexpected_fields = set(data.keys()) - set(rules.keys())
        if unexpected_fields:
            logger.warning(f"Unexpected fields in input: {unexpected_fields}")
        
        # If there are validation errors, raise them
        if errors:
            error_details = {
                'validation_errors': [e.to_dict() for e in errors],
                'field_count': len(errors)
            }
            raise ValidationError(
                f"Validation failed for {len(errors)} fields",
                severity=ErrorSeverity.HIGH,
                details=error_details
            )
        
        return validated_data
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        success_rate = 100.0
        if self.validation_stats['total_validations'] > 0:
            success_rate = (
                (self.validation_stats['total_validations'] - self.validation_stats['failed_validations']) /
                self.validation_stats['total_validations'] * 100
            )
        
        return {
            'total_validations': self.validation_stats['total_validations'],
            'failed_validations': self.validation_stats['failed_validations'],
            'success_rate': round(success_rate, 2),
            'validation_by_type': self.validation_stats['validation_by_type'].copy()
        }

# Global validator instance
input_validator = InputValidator()

# Convenience functions for common validations
def validate_symbol(symbol: str) -> str:
    """Validate stock symbol"""
    rule = ValidationRule(ValidationType.SYMBOL, required=True)
    return input_validator.validate_field(symbol, rule, "symbol")

def validate_price(price: Union[str, int, float]) -> Decimal:
    """Validate price"""
    rule = ValidationRule(ValidationType.PRICE, required=True, min_value=0.01, max_value=10000)
    return input_validator.validate_field(price, rule, "price")

def validate_quantity(quantity: Union[str, int]) -> int:
    """Validate share quantity"""
    rule = ValidationRule(ValidationType.QUANTITY, required=True, min_value=1, max_value=100000)
    return input_validator.validate_field(quantity, rule, "quantity")

def validate_trading_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate trading data with comprehensive rules"""
    rules = {
        'symbol': ValidationRule(ValidationType.SYMBOL, required=True),
        'action': ValidationRule(ValidationType.STRING, required=True, allowed_values=['BUY', 'SELL', 'HOLD']),
        'price': ValidationRule(ValidationType.PRICE, required=True, min_value=0.01, max_value=10000),
        'quantity': ValidationRule(ValidationType.QUANTITY, required=True, min_value=1, max_value=100000),
        'confidence': ValidationRule(ValidationType.PERCENTAGE, required=False, min_value=0, max_value=100),
        'date': ValidationRule(ValidationType.DATE, required=False)
    }
    
    return input_validator.validate_dict(data, rules)