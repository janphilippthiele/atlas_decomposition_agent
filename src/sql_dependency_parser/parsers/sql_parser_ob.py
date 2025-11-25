"""
SQL parser module for database dependency analysis.

This module provides object-scoped parsing to extract database objects
and their specific dependencies from SQL files.
"""

import re
import time
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import chardet
from tqdm import tqdm


class ObjectType(Enum):
    """Database object types."""
    TABLE = "TABLE"
    VIEW = "VIEW"
    PROCEDURE = "PROCEDURE"
    FUNCTION = "FUNCTION"
    PACKAGE = "PACKAGE"
    TRIGGER = "TRIGGER"
    SEQUENCE = "SEQUENCE"
    INDEX = "INDEX"
    SYNONYM = "SYNONYM"
    TYPE = "TYPE"
    PACKAGE_BODY = "PACKAGE_BODY"
    JOB = "JOB"
    CONSTRAINT = "CONSTRAINT"
    DATABASE_LINK = "DATABASE_LINK"
    USER_SCHEMA = "USER_SCHEMA"
    UNKNOWN = "UNKNOWN"
    MATERIALIZED_VIEW = "MATERIALIZED_VIEW"
    MATERIALIZED_VIEW_LOG = "MATERIALIZED_VIEW_LOG"

class PermissionType(Enum):
    """Types of permissions."""
    SELECT = "SELECT"
    INSERT = "INSERT" 
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    EXECUTE = "EXECUTE"
    ALL = "ALL"

class SecurityRelationType(Enum):
    """Types of security relationships."""
    OWNS = "OWNS"           # User owns object (derived from CREATE statements)
    GRANTS = "GRANTS"       # User grants permission to another user/role  
    HAS_PERMISSION = "HAS_PERMISSION"  # User/role has permission on object


class DependencyType(Enum):
    """Types of SQL dependencies."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    JOIN = "JOIN"
    MERGE = "MERGE"
    FOREIGN_KEY = "FOREIGN_KEY"
    CALL = "CALL"
    TRIGGER = "TRIGGER"
    EXECUTE = "EXECUTE"
    INHERITANCE = "INHERITANCE"
    ATTRIBUTE = "ATTRIBUTE"
    EXTERNAL_SYSTEM = "EXTERNAL_SYSTEM"
    ALIAS = "ALIAS"
    POINTS_TO = "POINTS_TO"
    CONNECTS_TO = "CONNECTS_TO"
    UNKNOWN = "UNKNOWN"

@dataclass
class SecurityRelationship:
    """Represents a security/permission relationship."""
    source_principal: str  # User or role name
    target_object: str     # Object name
    target_schema: Optional[str]
    relation_type: SecurityRelationType
    permissions: List[PermissionType] = field(default_factory=list)
    line_number: int = 0
    context: str = ""

@dataclass 
class ObjectDependency:
    """Represents a dependency from one object to another."""
    
    # Target object information
    target_name: str
    target_schema: Optional[str]
    target_object_type: ObjectType
    
    # Dependency details
    dependency_type: DependencyType
    line_number: int  # Line within the source object
    context: str  # The actual SQL line/statement
    


@dataclass
class DatabaseObject:
    """Complete definition of a database object with its dependencies."""
    
    # Basic identification
    name: str
    schema: Optional[str] 
    object_type: ObjectType
    
    # Location information
    file_path: str
    start_position: int
    end_position: int
    start_line: int
    end_line: int

    #Security relationships
    #security_relationships: List[SecurityRelationship] = field(default_factory=list)
    #owner: Optional[str] = None  # Derived from CREATE context or explicit ownership
    
    # Content
    full_content: str  # Complete object definition
    body_content: str  # Just the executable body (for procedures/functions)
    
    # Dependencies
    dependencies: List[ObjectDependency] = field(default_factory=list)
    
    @property
    def full_name(self) -> str:
        """Return fully qualified object name."""
        if self.schema:
            return f"{self.schema}.{self.name}"
        return self.name
    
    def __hash__(self):
        return hash((self.full_name.upper(), self.object_type))
    
    def __eq__(self, other):
        if not isinstance(other, DatabaseObject):
            return False
        return (self.full_name.upper() == other.full_name.upper() and 
                self.object_type == other.object_type)


@dataclass
class ParseResult:
    """Results from parsing a file."""
    file_path: Path
    objects_defined: List[DatabaseObject]  # Changed to List to preserve order
    errors: List[str]
    warnings: List[str]
    parse_time: float


class SQLParser:
    """
    Parser for SQL files to extract object-scoped database dependencies.
    
    This parser identifies individual database objects and analyzes their
    specific dependencies rather than file-level dependencies.
    """
    
    # Patterns for identifying object creation
    CREATE_PATTERNS = {
        'TABLE': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:GLOBAL\s+)?(?:TEMPORARY\s+)?TABLE\s+'
            r'(?:IF\s+NOT\s+EXISTS\s+)?("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
            re.IGNORECASE | re.MULTILINE
        ),
        'VIEW': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+FORCE\s)?(?:EDITIONABLE\s+)?(?:MATERIALIZED\s+)?VIEW\s+'
            r'(?:IF\s+NOT\s+EXISTS\s+)?("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
            re.IGNORECASE | re.MULTILINE
        ),
        'PROCEDURE': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+)?PROCEDURE\s+'
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?\.)?"?([a-zA-Z_][a-zA-Z0-9_$#]*)"?',
            re.IGNORECASE | re.MULTILINE
        ),
        'FUNCTION': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+)?FUNCTION\s+'
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?\.)?"?([a-zA-Z_][a-zA-Z0-9_$#]*)"?',
            re.IGNORECASE | re.MULTILINE
        ),
        'PACKAGE': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+)?PACKAGE\s+(?!BODY\s+)'
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?\.)?"?([a-zA-Z_][a-zA-Z0-9_$#]*)"?',
            re.IGNORECASE | re.MULTILINE
        ),
        'TRIGGER': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+)?TRIGGER\s+'
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?\.)?"?([a-zA-Z_][a-zA-Z0-9_$#]*)"?',
            re.IGNORECASE | re.MULTILINE
        ),
        'TYPE': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+)?TYPE\s+'
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?\.)?"?([a-zA-Z_][a-zA-Z0-9_$#]*)"?'
            r'(?:\s+(?:BODY|AS\s+TABLE\s+OF|UNDER))?',
            re.IGNORECASE | re.MULTILINE
        ),
        'PACKAGE_BODY': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:EDITIONABLE\s+)?PACKAGE\s+BODY\s+'
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?\.)?"?([a-zA-Z_][a-zA-Z0-9_$#]*)"?',
            re.IGNORECASE | re.MULTILINE
        ),
        'SEQUENCE': re.compile(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?SEQUENCE\s+'
            r'(?:("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\.)?'
            r'"?([a-zA-Z_][a-zA-Z0-9_$#]*)"?',
            re.IGNORECASE
        ),
        'INDEX': re.compile(
            r'CREATE\s+(?:UNIQUE\s+)?(?:BITMAP\s+)?INDEX\s+'
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\s+ON\s+'
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
            re.IGNORECASE | re.MULTILINE
        ),

        'CONSTRAINT': re.compile(
            # Match the full ALTER TABLE ... ADD CONSTRAINT structure
            # But capture constraint name parts in consistent groups
            r'(?:ALTER\s+TABLE\s+[^)]*\s+)?'  # Non-capturing: table part (optional)
            r'ADD\s+(?:CONSTRAINT\s+)?'        # Non-capturing: ADD CONSTRAINT keywords
            r'(?:("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\.)?' # Group 1: schema (optional)
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\s+'     # Group 2: constraint name
            r'(?:PRIMARY\s+KEY|UNIQUE|CHECK|FOREIGN\s+KEY|NOT\s+NULL)',
            re.IGNORECASE | re.MULTILINE
        ),

        'SYNONYM': re.compile(
            # CREATE SYNONYM captures the synonym name in standard groups
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:PUBLIC\s+)?SYNONYM\s+'
            r'(?:("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\.)?' # Group 1: schema (optional)
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\s+'     # Group 2: synonym name
            r'FOR\s+[^;]+',  # Non-capturing: everything after FOR
            re.IGNORECASE | re.MULTILINE
        ),

        'DATABASE_LINK': re.compile(
            # DATABASE LINK with consistent group structure
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:PUBLIC\s+)?DATABASE\s+LINK\s+'
            r'(?:("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\.)?' # Group 1: schema (optional, rarely used)
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)',       # Group 2: link name
            re.IGNORECASE | re.MULTILINE
        ),

        'REF_CONSTRAINT': re.compile(
            # Focus on capturing the constraint name in standard groups
            r'(?:ALTER\s+TABLE\s+[^)]*\s+)?'         # Non-capturing: table part
            r'ADD\s+(?:CONSTRAINT\s+)?'              # Non-capturing: keywords
            r'(?:("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\.)?' # Group 1: schema (optional)
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\s+'     # Group 2: constraint name
            r'FOREIGN\s+KEY\s*\([^)]+\)\s*REFERENCES\s+[^;]+', # Non-capturing: rest
            re.IGNORECASE | re.MULTILINE
        ),
}
    
    # Patterns for finding dependencies within object content
    DEPENDENCY_PATTERNS = {
        'TABLE': {
            'FOREIGN_KEY': re.compile(
                r'FOREIGN\s+KEY\s*\([^)]+\)\s*REFERENCES\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'SEQUENCE_DEFAULT': re.compile(
                r'DEFAULT\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\s*\.\s*NEXTVAL',
                re.IGNORECASE
            ),
            'CHECK_CONSTRAINT_FUNCTION': re.compile(
                r'CHECK\s*\([^)]*("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\s*\(',
                re.IGNORECASE
            )
        },
        
        'VIEW': {
            'SELECT_FROM': re.compile(
                r'(?<!DELETE\s)(?<!UPDATE\s)\bFROM\s+'  # Not preceded by DELETE or UPDATE
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'JOIN': re.compile(
                r'(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?'
                r'JOIN\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'FUNCTION_CALL': re.compile(
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\s*\(',
                re.IGNORECASE
            ),
            'MATERIALIZED_VIEW_LOG': re.compile(
                r'CREATE\s+MATERIALIZED\s+VIEW\s+LOG\s+ON\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'REFERENCES_SYNONYM': re.compile(
                r'(?:FROM|JOIN|INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'REFERENCES_DATABASE_LINK': re.compile(
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)@("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)',
                re.IGNORECASE
            ),
            'REFERENCES_MATERIALIZED_VIEW': re.compile(
                r'(?:FROM|JOIN|INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
        },
        
        'PROCEDURE': {
            'SELECT_FROM': re.compile(
                r'(?<!DELETE\s)(?<!UPDATE\s)\bFROM\s+'  # Not preceded by DELETE or UPDATE
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'INSERT_INTO': re.compile(
                r'INSERT\s+(?:IGNORE\s+)?INTO\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),    
            'UPDATE': re.compile(
                r'\bUPDATE\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?'  # Optional alias
                r'(?=\s+SET)',  # Must be followed by SET
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'DELETE_FROM': re.compile(
                r'DELETE\s+FROM\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+WHERE|\s*;|\s+SET)',  # Look ahead to confirm it's a table name
                re.IGNORECASE | re.MULTILINE
            ),
            'CALL_PROCEDURE': re.compile(
                r'\b(?:CALL|EXEC)\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'PACKAGE_CALL': re.compile(
                r'\b("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\.("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\s*\(',
                re.IGNORECASE
            ),
            'SEQUENCE_NEXTVAL': re.compile(
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\.\s*NEXTVAL',
                re.IGNORECASE
            ),
            'DML_OPERATIONS': re.compile(
                r'\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'MERGE_INTO': re.compile(
                r'\bMERGE\s+INTO\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'REFERENCES_SYNONYM': re.compile(
                r'(?:FROM|JOIN|INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'REFERENCES_DATABASE_LINK': re.compile(
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)@("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)',
                re.IGNORECASE
            ),
        },
        
        'FUNCTION': {
            # Similar to PROCEDURE but might have different patterns
            'SELECT_FROM': re.compile(
                r'(?<!DELETE\s)(?<!UPDATE\s)\bFROM\s+'  # Not preceded by DELETE or UPDATE
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'FUNCTION_CALL': re.compile(
                r'\b([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)\s*\(',
                re.IGNORECASE
            ),
            'REFERENCES_SYNONYM': re.compile(
                r'(?:FROM|JOIN|INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'REFERENCES_DATABASE_LINK': re.compile(
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)@("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)',
                re.IGNORECASE
            ),
        },
        
        'INDEX': {
            # Empty - we handle INDEX dependencies specially in object boundary detection
        },
        
        'TRIGGER': {
            'SELECT_FROM': re.compile(
                r'(?<!DELETE\s)(?<!UPDATE\s)\bFROM\s+'  # Not preceded by DELETE or UPDATE
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'INSERT_INTO': re.compile(
                r'INSERT\s+(?:IGNORE\s+)?INTO\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),    
            'UPDATE': re.compile(
                r'\bUPDATE\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?'  # Optional alias
                r'(?=\s+SET)',  # Must be followed by SET
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'DELETE_FROM': re.compile(
                r'DELETE\s+FROM\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+WHERE|\s*;|\s+SET)',  # Look ahead to confirm it's a table name
                re.IGNORECASE | re.MULTILINE
            ),
            'FUNCTION_CALL': re.compile(
                r'(?:=|:=|\|\||,)\s*([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)\s*\(',
                re.IGNORECASE
            ),
            'PACKAGE_CALL': re.compile(
                r'\b("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\.("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)\s*\(',
                re.IGNORECASE
            ),
            'TRIGGER_ON': re.compile(
                r'ON\s+([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)',
                re.IGNORECASE
            ),
            'REFERENCES_SYNONYM': re.compile(
                r'(?:FROM|JOIN|INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
        },

        'PACKAGE_BODY': {
            'SELECT_FROM': re.compile(
                r'(?<!DELETE\s)(?<!UPDATE\s)\bFROM\s+'  # Not preceded by DELETE or UPDATE
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'INSERT_INTO': re.compile(
                r'INSERT\s+(?:IGNORE\s+)?INTO\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),    
            'UPDATE': re.compile(
                r'\bUPDATE\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?'  # Optional alias
                r'(?=\s+SET)',  # Must be followed by SET
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'DELETE_FROM': re.compile(
                r'DELETE\s+FROM\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+WHERE|\s*;|\s+SET)',  # Look ahead to confirm it's a table name
                re.IGNORECASE | re.MULTILINE
            ),
            'CALL_PROCEDURE': re.compile(
                r'\b(?:CALL|EXEC)\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'SEQUENCE_NEXTVAL': re.compile(
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\.\s*NEXTVAL',
                re.IGNORECASE
            ),
            'MERGE_INTO': re.compile(
                r'\bMERGE\s+INTO\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?',  # Optional alias
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'FUNCTION_CALL': re.compile(
                r'(?:=|:=|\|\||,)\s*([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)\s*\(',
                re.IGNORECASE
            ),
            'SEQUENCE_NEXTVAL': re.compile(
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'\s*\.\s*NEXTVAL',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
        },
        'TYPE': {
            'TYPE_REFERENCE': re.compile(
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\%(?:TYPE|ROWTYPE)',
                re.IGNORECASE
            ),
            'FUNCTION_CALL': re.compile(
                r'(?:=|:=|\|\||,)\s*([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)\s*\(',
                re.IGNORECASE
            ),
            'MEMBER_FUNCTION': re.compile(
                r'MEMBER\s+(?:FUNCTION|PROCEDURE)\s+[a-zA-Z_][a-zA-Z0-9_$#]*.*?RETURN\s+([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)',
                re.IGNORECASE
            ),
            'TYPE_INHERITANCE': re.compile(
                r'TYPE\s+[^(]+\s+UNDER\s+([a-zA-Z_][a-zA-Z0-9_$#]*)',
                re.IGNORECASE | re.DOTALL),
            'TYPE_ATTRIBUTE': re.compile(
                r'^\s*[a-zA-Z_][a-zA-Z0-9_$#]*\s+([a-zA-Z_][a-zA-Z0-9_$#]*)',
                re.IGNORECASE)
        },

        'JOB': {
            'CALL_PROCEDURE': re.compile(
                r'(?:CALL|EXEC|EXECUTE)\s+([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)',
                re.IGNORECASE
            ),
            'PACKAGE_CALL': re.compile(
                r'([a-zA-Z_][a-zA-Z0-9_$#]*?)\.([a-zA-Z_][a-zA-Z0-9_$#]*?)\s*\(',
                re.IGNORECASE
            ),
            'SQL_BLOCK': re.compile(
                r'BEGIN\s+.*?END\s*;',
                re.IGNORECASE | re.DOTALL
            )
        },

        'CONSTRAINT': {
            'REFERENCES_TABLE': re.compile(
                r'FOREIGN\s+KEY\s*\([^)]+\)\s*REFERENCES\s+'
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'CHECK_FUNCTION': re.compile(
                r'CHECK\s*\([^)]*("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\s*\(',
                re.IGNORECASE
            ),
            'ON_TABLE': re.compile(
                r'(?:ALTER\s+TABLE|ADD\s+CONSTRAINT.*ON)\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'CHECK_SEQUENCE': re.compile(
                r'CHECK\s*\([^)]*("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\s*\.\s*(?:NEXTVAL|CURRVAL)',
                re.IGNORECASE
            )
        },

        'REF_CONSTRAINT': {
            'REFERENCES_TABLE': re.compile(
                r'REFERENCES\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'REFERENCES_CONSTRAINT': re.compile(
                r'REFERENCES\s+[^(]+\s*\([^)]+\)',
                re.IGNORECASE
            ),
            'ON_TABLE': re.compile(
                r'ALTER\s+TABLE\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            )
        },

        'SYNONYM': {
            'TARGET_OBJECT': re.compile(
                r'FOR\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'DATABASE_LINK': re.compile(
                r'FOR\s+[^@]+@("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)',
                re.IGNORECASE
            ),
        },

        'DATABASE_LINK': {
            'CONNECT_TO_USER': re.compile(
                r'CONNECT\s+TO\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)',
                re.IGNORECASE
            )
        },

        'MATERIALIZED_VIEW': {
            'SELECT_FROM': re.compile(
                r'(?<!DELETE\s)(?<!UPDATE\s)\bFROM\s+'  # Not preceded by DELETE or UPDATE
                r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)'
                r'(?:\s+[a-zA-Z_][a-zA-Z0-9_$#]*)?',
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            ),
            'JOIN': re.compile(
                r'(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?'
                r'JOIN\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            ),
            'REFRESH_ON': re.compile(
                r'REFRESH\s+(?:FAST|COMPLETE|FORCE)\s+ON\s+(?:COMMIT|DEMAND)',
                re.IGNORECASE
            ),
            'DATABASE_LINK_REFERENCE': re.compile(
                r'FROM\s+[^@]+@("?[a-zA-Z_][a-zA-Z0-9_$#]*"?)',
                re.IGNORECASE
            ),
            'FUNCTION_CALL': re.compile(
            r'("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)\s*\(',
            re.IGNORECASE
            ),
        },

        'MATERIALIZED_VIEW_LOG': {
            'SOURCE_TABLE': re.compile(
                r'ON\s+("?[a-zA-Z_][a-zA-Z0-9_$#]*"?(?:\."?[a-zA-Z_][a-zA-Z0-9_$#]*"?)?)',
                re.IGNORECASE
            )
        }
    }

    # For later security/permission analysis
    PERMISSION_PATTERNS = {
        'GRANT': re.compile(
            r'GRANT\s+([\w,\s]+)\s+ON\s+([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)\s+TO\s+([\w,\s]+)',
            re.IGNORECASE
        ),
        'REVOKE': re.compile(
            r'REVOKE\s+([\w,\s]+)\s+ON\s+([a-zA-Z_][a-zA-Z0-9_$#]*(?:\.[a-zA-Z_][a-zA-Z0-9_$#]*)?)\s+FROM\s+([\w,\s]+)',
            re.IGNORECASE
        ),
        'CREATE_USER': re.compile(
            r'CREATE\s+USER\s+([a-zA-Z_][a-zA-Z0-9_$#]*)',
            re.IGNORECASE
        ),
        'CREATE_ROLE': re.compile(
            r'CREATE\s+ROLE\s+([a-zA-Z_][a-zA-Z0-9_$#]*)',
            re.IGNORECASE
        ),
    }

    BUILTIN_FUNCTIONS = {
    'UPPER', 'LOWER', 'TRIM', 'LTRIM', 'RTRIM', 'SUBSTR', 'LENGTH',
    'TO_CHAR', 'TO_DATE', 'TO_NUMBER', 'NVL', 'NVL2', 'DECODE',
    'COALESCE', 'GREATEST', 'LEAST', 'ABS', 'ROUND', 'TRUNC', 'REPLACE',
    'REGEXP_REPLACE', 'REGEXP_SUBSTR', 'REGEXP_INSTR', 'REGEXP_COUNT',
    'ANARRAY', 'COLLECT', 'CHR', 'RANK', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
    'KEEP', 'FIRST_VALUE', 'LAST_VALUE', 'LEAD', 'LAG', 'LPAD', 'RPAD', 'CAST',
    'EXTRACT', 'TO_TIMESTAMP', 'NEXT', 'ADDDAYS', 'ADD_MONTHS', 'CONCAT', 'XMLTABLE',
    'NTILE'
    }

    BUILTIN_TYPES = {
        'VARCHAR2', 'NUMBER', 'DATE', 'CLOB', 'BLOB', 'CHAR', 'INTEGER',
        'NUMBERS', 'FLOAT', 'TIMESTAMP', 'INTERVAL', 'RAW', 'BOOLEAN'
    }

    BUILTIN_PACKAGES = {
    'DBMS_OUTPUT', 'DBMS_SQL', 'DBMS_LOB', 'DBMS_UTILITY', 
    'UTL_FILE', 'UTL_HTTP', 'DBMS_SCHEDULER','DBMS_CRYPTO', 'XMLTYPE'
    }

    SQL_KEYWORDS = {
    'FROM', 'WHERE', 'WHEN', 'WHILE', 'OR', 'RETURN', 'EXISTS', 
    'JOIN', 'THEN', 'SET', 'UNION', 'TABLE', 'ESCAPE', 'RECORD',
    'IF', 'SELECT', 'IMMEDIATE', 'IN', 'OVER', 'ORDER', 'AS', 
    'BY', 'PARTITION', 'GROUP', 'ROWS', 
    'RANGE', 'UNBOUNDED', 'PRECEDING', 'FOLLOWING', 'CURRENT', 
    'ROW', 'AND', 'IN', 'CURSOR', 'SELECT', 'FROM', 'WHERE', 'EXCEPTION_INIT'
    }

    ORACLE_TYPES = {
    'VARCHAR', 'ANYDATAS', 'STRINGS', 'LISTS', 'ARRAYS', 'OBJECTS',
    }

    ORACLE_BUILTIN_FUNCTIONS = { 
        'CEIL', 'TO_CLOB', 'LISTAGG', 'NLSSORT', 'RAWTOHEX', 'HEXTORAW',
        'INSTR', 'TO_DSINTERVAL', 'MOD', 'NUMTODSINTERVAL', 'SIGN',
        'DENSE_RANK', 'EXTEND', 'TRANSLATE'
    }

    
    # Comment patterns
    SINGLE_LINE_COMMENT = re.compile(r'--.*$', re.MULTILINE)
    MULTI_LINE_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
    
    # String literal patterns
    SINGLE_QUOTE_STRING = re.compile(r"'(?:[^']|'')*'")
    DOUBLE_QUOTE_STRING = re.compile(r'"(?:[^"]|"")*"')
    
    # CTE pattern
    WITH_CTE = re.compile(
        r'WITH\s+(?:RECURSIVE\s+)?([a-zA-Z_][a-zA-Z0-9_$#]*)\s+AS\s*\(',
        re.IGNORECASE
    )
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the parser with optional configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse a single file and extract object-scoped database dependencies.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            ParseResult containing extracted objects with their dependencies
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Read file with encoding detection
            content = self._read_file_with_encoding(file_path)
            
            # Remove comments while preserving string literals
            cleaned_content = self._remove_comments(content)
            
            # Extract object boundaries and content
            objects_defined = self._extract_defined_objects(cleaned_content, str(file_path))
            
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {str(e)}")
            errors.append(str(e))
            objects_defined = []
        
        parse_time = time.time() - start_time
        
        return ParseResult(
            file_path=file_path,
            objects_defined=objects_defined,
            errors=errors,
            warnings=warnings,
            parse_time=parse_time
        )

    def parse_files(self, file_paths: List[Path], 
                   show_progress: bool = True) -> Dict[Path, ParseResult]:
        """Parse multiple files with optional progress reporting."""
        results = {}
        
        iterator = tqdm(file_paths, desc="Parsing files") if show_progress else file_paths
        
        for file_path in iterator:
            if show_progress:
                iterator.set_postfix({'file': file_path.name})
            results[file_path] = self.parse_file(file_path)
            
        return results

    def _read_file_with_encoding(self, file_path: Path) -> str:
        """Read file content with automatic encoding detection."""
        # Try UTF-8 first (most common)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            pass
        
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding', 'latin-1')
        confidence = detected.get('confidence', 0)
        
        if confidence < 0.7:
            self.logger.warning(
                f"Low confidence ({confidence:.2f}) in encoding detection "
                f"for {file_path}. Detected: {encoding}"
            )
        
        # Try detected encoding, fall back to latin-1
        try:
            return raw_data.decode(encoding)
        except (UnicodeDecodeError, TypeError):
            self.logger.warning(f"Failed to decode {file_path} with {encoding}, using latin-1")
            return raw_data.decode('latin-1', errors='replace')

    def _remove_comments(self, content: str) -> str:
        """Remove comments from content while preserving string literals."""
        # First, identify all string literals to protect them
        strings = []
        for match in self.SINGLE_QUOTE_STRING.finditer(content):
            strings.append((match.start(), match.end()))
        for match in self.DOUBLE_QUOTE_STRING.finditer(content):
            strings.append((match.start(), match.end()))
        
        # Sort strings by position
        strings.sort()
        
        # Function to check if position is inside a string
        def in_string(pos: int) -> bool:
            for start, end in strings:
                if start <= pos < end:
                    return True
            return False
        
        # Remove multi-line comments
        result = content
        while True:
            match = self.MULTI_LINE_COMMENT.search(result)
            if not match:
                break
            if not in_string(match.start()):
                result = result[:match.start()] + ' ' + result[match.end():]
            else:
                break
        
        # Remove single-line comments
        lines = result.split('\n')
        cleaned_lines = []
        
        for line in lines:
            comment_pos = line.find('--')
            if comment_pos >= 0 and not in_string(comment_pos):
                line = line[:comment_pos]
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _normalize_object_name(self, name: str, schema: Optional[str] = None) -> Tuple[str, Optional[str]]:

        def normalize_identifier(identifier: str) -> str:
            if not identifier:
                return ""  # Return empty string instead of None
            
            # Clean the identifier
            identifier = identifier.strip()
            
            # Remove surrounding quotes if present
            if identifier.startswith('"') and identifier.endswith('"'):
                return identifier[1:-1]
            
            # Unquoted identifiers are case-insensitive (uppercase in most DBs)
            return identifier.upper()
        
        # Ensure we never pass None to normalize_identifier
        safe_name = name if name is not None else ""
        safe_schema = schema if schema is not None else None
        
        return normalize_identifier(safe_name), normalize_identifier(safe_schema) if safe_schema else None

    def _parse_qualified_name(self, qualified_name: str) -> Tuple[str, Optional[str]]:
        """Parse a potentially schema-qualified name."""
        if not qualified_name:
            return "", None
            
        # Remove surrounding whitespace only
        qualified_name = qualified_name.strip()
        
        # Handle quoted schemas and names properly
        # Pattern: "SCHEMA"."NAME" or SCHEMA.NAME or "SCHEMA".NAME or SCHEMA."NAME"
        if '.' in qualified_name:
            parts = qualified_name.split('.', 1)  # Split only on first dot
            schema_part = parts[0].strip()
            name_part = parts[1].strip()
            
            # Clean quotes from each part
            if schema_part.startswith('"') and schema_part.endswith('"'):
                schema_part = schema_part[1:-1]
            if name_part.startswith('"') and name_part.endswith('"'):
                name_part = name_part[1:-1]
                
            return name_part.upper(), schema_part.upper()
        else:
            # No schema, just object name
            name_part = qualified_name
            if name_part.startswith('"') and name_part.endswith('"'):
                name_part = name_part[1:-1]
            return name_part.upper(), None

    def _extract_defined_objects(self, content: str, file_path: str) -> List[DatabaseObject]:
        """Extract objects defined in the SQL file with object-scoped dependency analysis."""
        objects = []
        
        # Find all object boundaries
        object_boundaries = self._find_object_boundaries(content)
        
        # Process each object
        for boundary in object_boundaries:
            try:
                # Validate boundary data
                if not boundary.get('name'):
                    self.logger.warning(f"Skipping object with missing name in {file_path}")
                    continue

                # Extract object content
                obj_content = content[boundary['start']:boundary['end']]
                
                # Get object body (executable part)
                body_content = self._extract_object_body(obj_content, boundary['type'])
                
                dependencies = []
                 # Create DatabaseObject with safe type lookup
                name = boundary['name']
                schema = boundary['schema']   
                # Special handling for INDEX objects - add table dependency
                if boundary['type'] == 'index' and boundary.get('table_name'):
                    table_dependency = ObjectDependency(
                        target_name=boundary['table_name'],
                        target_schema=boundary['table_schema'],
                        target_object_type=ObjectType.TABLE,
                        dependency_type=DependencyType.SELECT,  # Index "reads" the table
                        line_number=1,
                        context=obj_content.split('\n')[0].strip()
                    )
                    dependencies.append(table_dependency)

                elif boundary['type'] == 'package_body':
                    # Append _BODY to make it distinguishable
                    name_with_suffix = f"{name}_BODY"
                    
                    # Add dependency to package specification (without _BODY)
                    package_spec_dependency = ObjectDependency(
                        target_name=name,  # Original name, points to PACKAGE
                        target_schema=schema,
                        target_object_type=ObjectType.PACKAGE,
                        dependency_type=DependencyType.EXECUTE,
                        line_number=1,
                        context="Package body implements package specification"
                    )
                    dependencies.append(package_spec_dependency)
                    
                    # Analyze other dependencies
                    dependencies.extend(
                        self._analyze_object_dependencies(body_content, boundary['type'], name, schema)
                    )
                    
                    # Use the modified name for the object itself
                    name = name_with_suffix

                else:
                    # Analyze dependencies within this object
                    dependencies = self._analyze_object_dependencies(body_content, boundary['type'], name, schema)

                         

                # Create DatabaseObject
                name, schema = self._normalize_object_name(boundary['name'], boundary['schema'])
                if boundary['type'] == 'PACKAGE_BODY':
                    name = f"{name}_BODY"
                
                obj = DatabaseObject(
                    name=name,
                    schema=schema,
                    object_type=getattr(ObjectType, boundary['type'].upper()),
                    file_path=file_path,
                    start_position=boundary['start'],
                    end_position=boundary['end'],
                    start_line=content[:boundary['start']].count('\n') + 1,
                    end_line=content[:boundary['end']].count('\n') + 1,
                    full_content=obj_content,
                    body_content=body_content,
                    dependencies=dependencies
                )
                
                objects.append(obj)
                    
            except Exception as e:
                self.logger.warning(f"Error processing object {boundary['name']}: {e}")
                continue
        
        return objects

    def _find_object_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """Find boundaries of all database objects in the content."""
        boundaries = []
        
        for obj_type, pattern in self.CREATE_PATTERNS.items():
            for match in pattern.finditer(content):
                start_pos = match.start()
                
                # Extract object name and schema
                if obj_type == 'INDEX':
                    # INDEX pattern has: group(1) = index name, group(2) = table name
                    index_name_raw = match.group(1)
                    table_name_raw = match.group(2) if match.lastindex >= 2 else None
                    
                    name, schema = self._parse_qualified_name(index_name_raw)
                    
                    # Store table info for dependency analysis
                    table_name, table_schema = self._parse_qualified_name(table_name_raw) if table_name_raw else (None, None)
            
                
                elif obj_type in ['PROCEDURE', 'FUNCTION', 'PACKAGE', 'TRIGGER', 'TYPE', 'PACKAGE_BODY', 'SEQUENCE']:
                    # These patterns have schema in group 1, name in group 2
                    schema_raw = match.group(1)
                    name_raw = match.group(2)
                    
                    if schema_raw:
                        schema_raw = schema_raw.rstrip('.').strip('"')
                        schema = schema_raw.upper() if schema_raw else None
                    else:
                        schema = None
                        
                    name = name_raw.strip('"').upper() if name_raw else ""
                    
                else:
                    # TABLE, VIEW, etc. have full name in group 1
                    full_name = match.group(1)
                    name, schema = self._parse_qualified_name(full_name)
                
                # Validate that we have a proper name
                if not name:
                    continue
                
                # Find object end
                end_pos = self._find_object_end(content, start_pos, obj_type)
                
                if end_pos > start_pos:
                    boundary_info = {
                        'type': obj_type.lower(),
                        'name': name,
                        'schema': schema,
                        'start': start_pos,
                        'end': end_pos
                    }
                    
                    # Add table info for INDEX objects
                    if obj_type == 'INDEX' and table_name:
                        boundary_info['table_name'] = table_name
                        boundary_info['table_schema'] = table_schema
                    
                    boundaries.append(boundary_info)
        
        # Sort by start position to handle overlaps
        boundaries.sort(key=lambda x: x['start'])
        
        return boundaries


    def _find_object_end(self, content: str, start_pos: int, obj_type: str) -> int:
        """Find the end position of a database object."""
        remaining_content = content[start_pos:]
        
        if obj_type in ['PROCEDURE', 'FUNCTION', 'PACKAGE', 'TRIGGER']:
            # Find matching END statement
            end_pattern = re.compile(r'\bEND\s*(?:[a-zA-Z_][a-zA-Z0-9_$#]*)?\s*;', re.IGNORECASE)
            
            # Simple approach: find first END; after the object start
            # This could be improved to handle nested blocks
            match = end_pattern.search(remaining_content)
            if match:
                return start_pos + match.end()
            else:
                # Fallback: end of file
                return len(content)
                
        elif obj_type in ['TABLE', 'VIEW']:
            # Find next semicolon
            semicolon_pos = remaining_content.find(';')
            if semicolon_pos >= 0:
                return start_pos + semicolon_pos + 1
            else:
                return len(content)
        
        elif obj_type == 'JOB':
            # Find the closing parenthesis of the procedure call
            paren_count = 0
            in_string = False
            for i, char in enumerate(remaining_content):
                if char == "'" and (i == 0 or remaining_content[i-1] != '\\'):
                    in_string = not in_string
                elif not in_string:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            # Find the semicolon after this
                            semicolon_pos = remaining_content.find(';', i)
                            return start_pos + (semicolon_pos + 1 if semicolon_pos >= 0 else i + 1)
            return len(content)
        
        return len(content)

    def _extract_object_body(self, obj_content: str, obj_type: str) -> str:
        """Extract the executable body of an object."""
        if obj_type in ['procedure', 'function', 'package', 'trigger']:
            # Find the body between IS/AS and END
            body_start_pattern = re.compile(r'\b(?:IS|AS)\b', re.IGNORECASE)
            match = body_start_pattern.search(obj_content)
            if match:
                body_start = match.end()
                # Find the last END statement
                end_pattern = re.compile(r'\bEND\b.*?;', re.IGNORECASE)
                end_matches = list(end_pattern.finditer(obj_content))
                if end_matches:
                    body_end = end_matches[-1].start()
                    return obj_content[body_start:body_end].strip()
            
            # Fallback: return everything after CREATE line
            lines = obj_content.split('\n')
            if len(lines) > 1:
                return '\n'.join(lines[1:]).strip()
                
        elif obj_type == 'view':
            # Find the SELECT statement
            as_pattern = re.compile(r'\bAS\s+', re.IGNORECASE)
            match = as_pattern.search(obj_content)
            if match:
                return obj_content[match.end():].strip()
        
        # For tables or fallback, return the full content
        return obj_content

    def _analyze_object_dependencies(self, body_content: str, obj_type: str, source_name: str, source_schema: str) -> List[ObjectDependency]:
        """Analyze dependencies within a specific object's body content."""
        dependencies = []

        # Get patterns for this object type
        type_patterns = self.DEPENDENCY_PATTERNS.get(obj_type.upper(), {})
        
        # Extract CTE names to avoid false positives
        cte_names = self._extract_cte_names(body_content)
        
        # Find dependencies using each pattern
        for dep_type, pattern in type_patterns.items():
            for match in pattern.finditer(body_content):
                target_name_raw = match.group(1)

                name, schema = self._parse_qualified_name(target_name_raw)
            
                # Skip built-in functions and types
                if (name.upper() in self.BUILTIN_FUNCTIONS or
                    name.upper() in self.BUILTIN_TYPES or
                    name.upper() in self.BUILTIN_PACKAGES or
                    name.upper() in self.SQL_KEYWORDS or
                    name.upper() in self.ORACLE_TYPES or
                    name.upper() in self.ORACLE_BUILTIN_FUNCTIONS):
                    continue
                
                if name.startswith('L_') or len(name) <= 2:
                    continue
                # Skip string literals (contain dots or quotes)
                if '.' in target_name_raw and not schema:  # Like "java.sql.Blob"
                    continue
                

                # Check self-references
                if (schema == source_schema and name == source_name) or \
                    (name == source_name):
                    continue
                                    
                # Skip CTEs
                if target_name_raw.upper() in cte_names:
                    continue
                
                if schema:
                    if schema.upper() in self.BUILTIN_PACKAGES or schema.upper() == 'SELF':
                        continue
                  
                # Determine target object type and dependency type
                target_obj_type, dependency_type = self._infer_target_type_and_dependency(dep_type)

                # Get line number within the object body
                line_number = body_content[:match.start()].count('\n') + 1
                
                # Get context (the line containing the match) and skip if its a java ref
                lines = body_content.split('\n')
                context_line = lines[line_number - 1] if line_number <= len(lines) else ""
                if any(keyword in context_line.upper() for keyword in ['NAME ', 'JAVA.', "'", '"']):
                    continue

                
                dependency = ObjectDependency(
                    target_name=name,
                    target_schema=schema,
                    target_object_type=target_obj_type,
                    dependency_type=dependency_type,
                    line_number=line_number,
                    context=context_line.strip()
                )
                
                dependencies.append(dependency)
        
        return dependencies

    def _infer_target_type_and_dependency(self, pattern_type: str) -> Tuple[ObjectType, DependencyType]:
        """Infer target object type and dependency type from pattern type."""
        type_mapping = {
            'SELECT_FROM': (ObjectType.TABLE, DependencyType.SELECT),
            'JOIN': (ObjectType.TABLE, DependencyType.JOIN),
            'INSERT_INTO': (ObjectType.TABLE, DependencyType.INSERT),
            'UPDATE': (ObjectType.TABLE, DependencyType.UPDATE),
            'DELETE_FROM': (ObjectType.TABLE, DependencyType.DELETE),
            'MERGE_INTO': (ObjectType.TABLE, DependencyType.MERGE),
            'CALL_PROCEDURE': (ObjectType.PROCEDURE, DependencyType.CALL),
            'FOREIGN_KEY': (ObjectType.TABLE, DependencyType.FOREIGN_KEY),
            'MATERIALIZED_VIEW_LOG': (ObjectType.TABLE, DependencyType.SELECT),  # Materialized view log
            # PL/SQL specific patterns
            'SEQUENCE_NEXTVAL': (ObjectType.SEQUENCE, DependencyType.SELECT),  # Reading sequence
            'SEQUENCE_CURRVAL': (ObjectType.SEQUENCE, DependencyType.SELECT),  # Reading sequence
            'PACKAGE_CALL': (ObjectType.PACKAGE, DependencyType.CALL),
            'TYPE_REFERENCE': (ObjectType.TYPE, DependencyType.SELECT),  # Using type definition
            'CURSOR_FOR': (ObjectType.TABLE, DependencyType.SELECT),
            'FUNCTION_INDEX': (ObjectType.FUNCTION, DependencyType.CALL),
            'FUNCTION_CALL': (ObjectType.FUNCTION, DependencyType.CALL),
            'TRIGGER_ON': (ObjectType.TABLE, DependencyType.TRIGGER),  # Trigger on table
            'MEMBER_FUNCTION': (ObjectType.TYPE, DependencyType.CALL),
            'TYPE_INHERITANCE': (ObjectType.TYPE, DependencyType.INHERITANCE),
            'TYPE_ATTRIBUTE': (ObjectType.TYPE, DependencyType.ATTRIBUTE),
            'REFERENCES_TABLE': (ObjectType.TABLE, DependencyType.FOREIGN_KEY),
            'CHECK_FUNCTION': (ObjectType.FUNCTION, DependencyType.CALL),
            'CHECK_SEQUENCE': (ObjectType.SEQUENCE, DependencyType.CALL),
            'REFERENCES_TABLE': (ObjectType.TABLE, DependencyType.FOREIGN_KEY),
            'REFERENCES_CONSTRAINT': (ObjectType.CONSTRAINT, DependencyType.FOREIGN_KEY),
            'TARGET_OBJECT': (ObjectType.UNKNOWN, DependencyType.ALIAS),
            'DATABASE_LINK': (ObjectType.DATABASE_LINK, DependencyType.POINTS_TO),
            'CONNECT_TO': (ObjectType.USER_SCHEMA, DependencyType.CONNECTS_TO),
            'DATABASE_LINK_REF': (ObjectType.DATABASE_LINK, DependencyType.CONNECTS_TO),
            'SOURCE_TABLE': (ObjectType.TABLE, DependencyType.SELECT)
        }
        
        return type_mapping.get(pattern_type, (ObjectType.UNKNOWN, DependencyType.UNKNOWN))

    def _extract_cte_names(self, content: str) -> Set[str]:
        """Extract CTE names to avoid false positive dependencies."""
        cte_names = set()
        
        for match in self.WITH_CTE.finditer(content):
            cte_name = match.group(1)
            cte_names.add(cte_name.upper())
            
            # Find additional CTEs in the same WITH clause
            remaining = content[match.end():]
            additional_cte_pattern = re.compile(
                r',\s*([a-zA-Z_][a-zA-Z0-9_$#]*)\s+AS\s*\(',
                re.IGNORECASE
            )
            for additional_match in additional_cte_pattern.finditer(remaining):
                cte_names.add(additional_match.group(1).upper())
                
        return cte_names

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return ['.sql', '.ddl', '.dml']

    def validate_config(self) -> List[str]:
        """Validate parser configuration."""
        errors = []
        
        if 'patterns' in self.config:
            for pattern_name, pattern_str in self.config['patterns'].items():
                try:
                    re.compile(pattern_str)
                except re.error as e:
                    errors.append(f"Invalid regex pattern '{pattern_name}': {str(e)}")
        
        return errors