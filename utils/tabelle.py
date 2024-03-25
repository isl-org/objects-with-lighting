# SPDX-License-Identifier: Apache-2.0
from typing import Any, Tuple, Optional, Literal, Union
from dataclasses import dataclass
import numpy as np
import numbers
from collections import namedtuple
from termcolor import colored
import warnings
import itertools
import re


__all__ = ['Cell', 'ColumnFormat', 'RowFormat', 'Table']

def _latex_to_unicode(s: str):
    from pylatexenc.latex2text import LatexNodes2Text
    return LatexNodes2Text().latex_to_text(s)    

@dataclass
class ColumnFormat:
    """Format options for columns in a table.
    
    This object should only be used in the cells in the first row of the table.
    
    Args:
        align: The alignment for the column. One of l,c,r.
        num_format: The number format, e.g. '{:>.2f}'. This will be translated 
            to the siunitx S column type and the table-format option.
        min_width: The minimum width for the column. This option only affects
            the terminal.
        auto_highlight: Automatically highlight the 'largest' or the 'smallest'
            values if set by making them bold or underlined.
        num_highlight: Defines how many values to highlight. At most two values
            can be highlighted. Values larger than 2 have no effect.
        line: Describes the vertical line right to the cell. Set to '' for no 
            vertical line (default). Set to '|' for a single vertical line and
            set to '||' for a double vertical line.
        leftmost_line: Describes the vertical line left to the cell. This is 
            only valid for the first column in the table.
    """
    align: str = 'c'
    num_format: str = '{}'
    min_width: Optional[int] = None
    auto_highlight: Literal['largest', 'smallest', None] = None
    num_highlight: int = 1
    line: Literal['', '|', '||'] = ''
    leftmost_line: Literal['', '|', '||'] = ''

@dataclass
class RowFormat:
    """Format options for rows in a table.
    
    This object should only be used in the cells in the first column of the table.

    Args:
        line: Describes the horizontal line below the cell. Set to True to 
            define a line spanning the entire row or set to a list of tuples
            with each tuple describing a segment. [(0,0), (1,2)] describes two
            horizontal lines. The first one spans the first column and the
            second one spans the second and third column.
        topmost_line: Describes the horizontal line above the cell. This is 
            only valid for the first row in the table.
    """
    auto_highlight: Literal['largest', 'smallest', None] = None # not implemented
    num_highlight: int = 1 # not implemented
    line: Union[bool,Tuple[Tuple[int,int]]] = False
    topmost_line: bool = False


@dataclass
class Cell:
    """Cell class that stores the value of the cell as well as format attributes
    
    Args:
        value: The value that will be printed. This can be any object that can be converted to a string.
        bold: If True make the text bold.
        underline: If True underline the text.
        col_span: The number of columns this cells spans.
        row_span: The number of rows this cells spans.
        col_format: Format options for the column. This should be None if the cell is not in the first row of the table.
        row_format: Format options for the row. This should be None if the cell is not in the first column of the table.
        table_format: Format options for the table. This should be None if the cell is not the first cell of the table.
    """
    value: Any
    bold: bool = False
    underline: bool = False
    col_span: int = 1
    row_span: int = 1
    col_format: Optional[ColumnFormat] = None
    row_format: Optional[RowFormat] = None

    @property
    def colfmt(self) -> ColumnFormat:
        """A convenience accessor for the col_format attribute.
        
        Accessing col_format with the colfmt property will initialize the attribute
        and should only be used on cells in the first row of the table.
        """
        if self.col_format is None:
            self.col_format = ColumnFormat()
        return self.col_format

    @colfmt.setter
    def colfmt(self, x):
        self.col_format = x
    
    @property
    def rowfmt(self) -> RowFormat:
        """A convenience accessor for the row_format attribute.
        
        Accessing row_format with the rowfmt property will initialize the attribute
        and should only be used on cells in the first column of the table.
        """
        if self.row_format is None:
            self.row_format = RowFormat()
        return self.row_format

    @rowfmt.setter
    def rowfmt(self, x):
        self.row_format = x
    
    
    def is_number(self):
        return isinstance(self.value, numbers.Number)

    def number_after_format(self, table, row: int, col: int):
        """Returns the number after applying the number format of the column.
        Returns None if the cell value is not a number.
        
        Args:
            table: The Table object.
            row: The row position of this cell
            col: The col position of this cell
        """
        column_format = table._get_cell(0,col).col_format
        if self.is_number():
            if column_format is not None:
                return float(column_format.num_format.format(self.value))
            else:
                return float(self.value)
        else:
            return None

    def can_convert_to_value(self):
        """Returns True if the cell can be safely converted to the value, i.e., 
        none of the format options have been changed."""
        for key in Cell.__dataclass_fields__:
            if key == 'value':
                continue
            if getattr(Cell,key) != getattr(self,key):
                return False
        return True

    def _base_str(self, table, row: int, col:int, line: Optional[int]=None):
        """Returns the string of value after applying the column number format.
        
        Args:
            table: The Table object.
            row: The row position of this cell
            col: The col position of this cell
            line: The line index for multiline values. Only used for multirow cells.
        Returns:
            Returns the string of value after applying the column number format.
        """
        column_format = table._get_cell(0,col).col_format

        if self.row_span > 1:
            value = self.value[line]
        else:
            value = self.value
        if self.is_number() and column_format is not None:
            s = column_format.num_format.format(value)
        elif value is None:
            s = ''
        else:
            s = str(value)
        if table._target == 'terminal' and table._convert_latex_to_unicode and table._unicode:
            s = _latex_to_unicode(s)
        return s


    def width(self, table, row: int, col: int):
        """Computes the width of the cell.
        Args:
            table: The Table object.
            row: The row position of this cell
            col: The col position of this cell
        Returns:
            The width of the cell
        """

        column_format = table._get_cell(0,col).col_format
        
        w = 0
        if column_format is not None and column_format.min_width is not None:
            w = column_format.min_width

        if self.row_span > 1:
            sw = 0
            for line in range(self.row_span):
                sw = max(sw,len(self._base_str(table, row, col, line=line)))
        else:
            sw = len(self._base_str(table, row, col))

        return max(w,sw+1) # +1 makes it look better


    def apply_attributes(self, s, attrs):
        """Apply terminal attributes to the string."""
        center = s.strip()
        left_spaces = len(s)-len(s.lstrip())
        right_spaces = len(s)-len(s.lstrip())
        return s[:left_spaces] + colored(center, attrs=attrs) + s[left_spaces+len(center):]
        

    def to_str(self, table, row: int, col: int, width: int, bold=None, underline=None, line: Optional[int]=None):
        """Returns a string representation of the cell for printing in a terminal.

        Args:
            table: The Table object.
            row: The row position of this cell
            col: The col position of this cell
            width: The target width.
            bold: Make the text bold even if the bold attribute of the cell is set to False.
            underline: Make the text underline even if the underline attribute of the cell is set to False.
            line: The line index for multiline values. Only used for multirow cells.
        
        Returns:
            Returns a string representation of the cell for printing in a terminal.
        """
        column_format = table._get_cell(0,col).col_format
        s = self._base_str(table, row, col,line=line)
        actual_len = len(s)
        attrs = []
        if bold or self.bold:
            attrs.append('bold')
        if underline or self.underline:
            attrs.append('underline')
        if attrs:
            s = self.apply_attributes(s, attrs=attrs)

        align = 'c'
        if column_format and column_format.align:
            align = column_format.align

        stralign = {'c':'^', 'l': '<', 'r': '>'}.get(align)
        format_specifier = '{:'+stralign+str(width+len(s)-actual_len)+'}'

        s = format_specifier.format(s)
        return s

    def to_latex(self, table, row: int, col: int, bold=None, underline=None):
        """Returns a latex representation of the cell.

        Args:
            table: The Table object.
            row: The row position of this cell
            col: The col position of this cell
            column_format: The column format.
            bold: Make the text bold even if the bold attribute of the cell is set to False.
            underline: Make the text underline even if the underline attribute of the cell is set to False.
        
        Returns:
            Returns a latex representation of the cell.
        """
        if self.row_span > 1:
            s = ''.join([self._base_str(table, row, col, line=i) for i in range(self.row_span)])
        else:
            s = self._base_str(table, row, col)
        if not self.is_number() and s != '':
            # wrap in {}
            s = '{'+s+'}'
        if (underline or self.underline) and s != '':
            s = r'\uline{' + s.strip() + '}'
        if (bold or self.bold) and s != '':
            s = r'\bfseries ' + s
        return s

        
# characters for drawing vertical and horizontal lines in the terminal
BoxDrawingStrings =  namedtuple('BoxDrawingStrings', ['NO_VLINE', 'LIGHT_VLINE', 'HEAVY_VLINE', 'LIGHT_HLINE', 'HEAVY_HLINE', 'LIGHT_CROSS', 'HEAVY_CROSS', 'LIGHT_VERT_LEFT', 'HEAVY_VERT_LEFT', 'LIGHT_VERT_RIGHT', 'HEAVY_VERT_RIGHT', 'LIGHT_LEFT', 'LIGHT_RIGHT'])

class Table:

    def __init__(self, shape: Tuple[int,int], unicode: bool=True, convert_latex_to_unicode=True) -> None:
        r"""Creates a table that can be printed to the terminal or converted to latex.
        
        The following example creates a booktabs-style table::
            from table import *

            tab = Table((6,5))
            tab[0,0].rowfmt.topmost_line = True # sets the top rule
            tab[0,0].colfmt.line = '|' # sets a vertical line, try to avoid for booktabs-style tables
            tab[0,1] = Cell('Quality', col_span=2, bold=True) # Use the Cell ctor to define the value of a cell and additional attributes
            tab[0,3] = Cell('Speed', col_span=2)
            tab[0,0].rowfmt.line = [(1,2),(3,4)] # horizontal line below row 0
            tab[1,:] = ['Method', r'MSE \downarrow', r'PSNR \uparrow', r'Velocity $\omega$', r'$\Delta t$']
            tab[1,0].rowfmt.line = True
            tab[2:,0] = ['A', 'B', 'C', 'D' ]
            tab[2:,1:] = np.random.rand(4,4)
            tab[-1,0].rowfmt.line = True # sets the bottom rule

            # setup format for the columns
            tab[0,1].colfmt.auto_highlight = 'smallest'
            tab[0,1].colfmt.num_highlight = 2
            tab[0,1].colfmt.num_format = '{:1.3f}'

            tab[0,2].colfmt.auto_highlight = 'largest'
            tab[0,2].colfmt.num_highlight = 1
            tab[0,2].colfmt.num_format = '{:1.3f}'

            tab[0,3].colfmt.num_format = '{:1.4f}'
            tab[0,4].colfmt.num_format = '{:1.4f}'

            print(tab.str())
            print(tab.latex(True, True))
        
        Args:
            shape: The rows and columns of the table as tuple (rows, cols).
            unicode: If True use unicode box drawing characters.
            convert_latex_to_unicode: If True convert latex to unicode symbols.
        """
        assert len(shape) == 2
        self._shape = shape
        self.clear()
        self._unicode = unicode
        self._convert_latex_to_unicode = convert_latex_to_unicode
        self._target = None
        if unicode:
            self.bds = BoxDrawingStrings('  ', ' \u2502 ', ' \u2551 ', '\u2500', '\u2501', '\u253c', '\u256b', '\u2524', '\u2563', '\u251c', '\u2560', '\u2574', '\u2576')
        else:
            self.bds = BoxDrawingStrings('  ', ' | ', ' || ', '-', '=', '+', '++', '+', '++', '+', '++', '-', '-')
        
        self.ltype2str = {'': self.bds.NO_VLINE, '|': self.bds.LIGHT_VLINE, '||': self.bds.HEAVY_VLINE}

    @property
    def shape(self) -> Tuple[int,int]:
        return self._shape

    def clear(self):
        """Clears the table values and all format information."""
        self._values = np.empty(self._shape, dtype=object)
        with np.nditer(self._values, flags=['refs_ok'], op_flags=['readwrite']) as it:
            for x in it:
                if not isinstance(x.item(),Cell):
                    x[...] = Cell(x.item())


    def __getitem__(self, key) -> Cell:
        return self._values[key]

    def __setitem__(self, key, value) -> None:
        tmp = np.asarray(value, dtype=object)
        if tmp.shape == tuple() and isinstance(self._values[key], Cell):
            if isinstance(tmp.item(), Cell):
                self._values[key] = tmp.item()
            else:
                self._values[key].value = tmp.item()
        else:
            tmp = np.broadcast_to(tmp, self._values[key].shape)
            with np.nditer(tmp, flags=['refs_ok']) as src_it:
                with np.nditer(self._values[key], flags=['refs_ok'], op_flags=['readwrite']) as dst_it:
                    for src, dst in zip(src_it, dst_it):
                        if isinstance(src.item(),Cell):
                            dst[...] = src.item()
                        else:
                            dst.item().value = src.item()

    def set_topbottomrule(self, value: bool):
        """Convenience function for setting the top and bottom rule"""
        self._get_cell(0,0).rowfmt.topmost_line = value
        self._get_cell(self._shape[0]-1,0).rowfmt.line = value

    def numpy(self, keep_cells_with_format_information: bool=False):
        """Returns a numpy array with dtype object and containing only the values of each cell.

        Args:
            keep_cells_with_format_information: If True return the Cell object if it stores 
                format information.
        
        Returns:
            numpy object array
        """
        # convert everything that can be converted to the original type
        values = self._values.copy()
        with np.nditer(values, flags=['refs_ok'], op_flags=['readwrite']) as it:
            for x in it:
                if x.item().can_convert_to_value() and not keep_cells_with_format_information:
                    x[...] = x.item().value
        return values

    def _get_column_format(self, col: int):
        """Helper to get the column format for a column.

        Args:
            col: The index of the column.
        
        Returns:
            The vertical line type.
        """
        format_ = self._get_cell(0,col).col_format
        if format_ is None:
            format_ = ColumnFormat()
        return format_

    def _get_row_format(self, row: int):
        """Helper to get the column format for a column.

        Args:
            col: The index of the column.
        
        Returns:
            The vertical line type.
        """
        format_ = self._get_cell(row,0).row_format
        if format_ is None:
            format_ = RowFormat()
        return format_


    def _get_vertical_line(self, left_col_idx: int):
        """Helper to get the vertical line for a column.

        Args:
            left_col_idx: The index of the column left to the vertical line. 
                Supports -1 to get the leftmost vertical line.
        
        Returns:
            The vertical line type.
        """
        format_ = self._get_column_format(max(0,left_col_idx))
        return format_.leftmost_line if left_col_idx == -1 else format_.line

    def _get_horizontal_line(self, top_row_idx: int):
        """Helper to get the horizontal line for a row.

        Args:
            top_row_idx: The index of the row to the top of the horizontal line. 
                Supports -1 to get the topmost vertical line.
        
        Returns:
            The horizontal line type.
        """
        format_ = self._get_row_format(max(0,top_row_idx))
        return format_.topmost_line if top_row_idx == -1 else format_.line


    def _get_cell(self, row, col):
        """Helper to get a cell as Cell object."""
        c = self._values[row,col]
        if isinstance(c, Cell):
            return c
        else:
            return Cell(c)


    def _extreme_values(self):
        """Helper for retrieving the extreme values for each column and row.
        
        Returns:
            A dictionay of the form {'cols':[a,b,c,...], 'rows':[a,b,c,...]}.
            Each column/row is a dictionary of the following form
            {'smallest': {
                1.23 : {'bold': True},
                99.9: {'underline': True},
                },
             'largest': {
                1.23 : {'underline': True},
                99.9: {'bold': True},
                }
            }
        """
        value_cell_map = self._value_cell_map()
        result = {}
        extreme_values = []
        for col in range(self._values.shape[1]):
            col_format = self._get_column_format(col)
            vals = []
            for row in range(self._values.shape[0]):
                cell = self._get_cell(*value_cell_map[row, col])
                if cell.is_number():
                    vals.append(cell.number_after_format(self, row, col))
            if col_format is not None and len(vals):
                vals.sort()
                smallest_dict = {val: {arg: True} for val, arg in zip(vals[:col_format.num_highlight], ('bold','underline'))}
                largest_dict = {val: {arg: True} for val, arg in zip(vals[::-1][:col_format.num_highlight], ('bold','underline'))}
                extreme_values.append({'largest': largest_dict, 'smallest': smallest_dict})
            else:
                extreme_values.append(None)
        result['cols'] = extreme_values

        extreme_values = []
        for row in range(self._values.shape[0]):
            row_format = self._get_row_format(row)
            vals = []
            for col in range(self._values.shape[1]):
                col_format = self._get_column_format(col)
                cell = self._get_cell(*value_cell_map[row, col])
                if cell.is_number():
                    vals.append(cell.number_after_format(self, row, col))
            if len(vals):
                vals.sort()
                smallest_dict = {val: {arg: True} for val, arg in zip(vals[:row_format.num_highlight], ('bold','underline'))}
                largest_dict = {val: {arg: True} for val, arg in zip(vals[::-1][:row_format.num_highlight], ('bold','underline'))}
                extreme_values.append({'largest': largest_dict, 'smallest': smallest_dict})
            else:
                extreme_values.append(None)
        result['rows'] = extreme_values

        return result




    def _value_cell_map(self):
        """This returns a table which maps cells to the coordinates of visible cells.
        
        Multicolumn or multirow cells hide their neighboring cells. This map 
        stores the coordinates of the cells that are actually visible at this
        location. If there are no multicolumn or multirow cells then all cells
        point to themself.
        """
        value_cell_map = np.empty_like(self._values)
        for row, col in itertools.product(*map(range,value_cell_map.shape)):
            cell = self._get_cell(row, col)
            for r,c in itertools.product(range(row,row+cell.row_span), range(col,col+cell.col_span)):
                if value_cell_map[r,c] is None:
                    value_cell_map[r,c] = (row,col)
        return value_cell_map


    def _check(self):
        """Do some sanity checks and print warnings."""
        value_cell_map = self._value_cell_map()

        for row in range(self._values.shape[0]):
            for col in range(self._values.shape[1]):
                cell = self._get_cell(row, col)
                if cell.col_format is not None and row > 0:
                    warnings.warn(f"Cell at {row,col} has defined col_format, which will be ignored.", stacklevel=3)

                if cell.row_format is not None and col > 0:
                    warnings.warn(f"Cell at {row,col} has defined row_format, which will be ignored.", stacklevel=3)

                if cell.value is not None and value_cell_map[row,col] != (row,col):
                    warnings.warn(f"Cell at {row,col} contains data that is hidden by a cell spanning multiple columns or rows at {value_cell_map[row,col]}.", stacklevel=3)

        return True
        

    def __str__(self) -> str:
        self._check()
        self._target = 'terminal'
        lines = []

        extreme_values = self._extreme_values()
        value_cell_map = self._value_cell_map()

        def compute_column_width(i, ignore_multi_col_cells=False) -> int:
            """Helper computing the width for column i"""
            w = 0
            for row in range(self._values.shape[0]):
                cell = self._get_cell(row,i)
                if cell.col_span == 1:
                    w = max(w, cell.width(self, row, i))

            if not ignore_multi_col_cells:
                for row in range(self._values.shape[0]):
                    cell = self._get_cell(row,i)
                    if cell.col_span > 1:
                        col_span_w = sum([w]+[compute_column_width(j, True) for j in range(i+1,i+cell.col_span)])
                        cell_w = cell.width(self,row,i)
                        if col_span_w < cell_w:
                            w += cell_w - col_span_w
            return w
        column_widths = [compute_column_width(i) for i in range(self._values.shape[1])]


        def compute_cell_width(cell, col):
            """Helper computing the width of a cell"""
            w = sum(column_widths[col:col+cell.col_span])
            for j in range(col, col+cell.col_span-1):
                vertical_line_type = self._get_vertical_line(j)
                # if vertical_line_type:
                vertical_line = self.ltype2str[vertical_line_type]
                w += len(vertical_line)
            return w


        def horizontal_line_str(row):
            """This function generates the horizontal lines."""
            horizontal_lines = self._get_horizontal_line(row)
            if not horizontal_lines:
                return None
            if row == -1 or row == self._values.shape[0]-1:
                line_char = self.bds.HEAVY_HLINE
            else:
                line_char = self.bds.LIGHT_HLINE

            # The key format is 
            # ( vertical_line_type, 
            #   is there a horizontal line to the left of the vertical line?,
            #   is there a horizontal line to the right of the vertical line?,
            #   do the left and right horizontal lines stem from the same segment?
            # )
            crossings = {
                ('', True, True, True): len(self.bds.NO_VLINE)*line_char,
                ('', True, True, False): self.bds.LIGHT_LEFT+self.bds.LIGHT_RIGHT,
                ('', False, True, False): ' '+self.bds.LIGHT_RIGHT,
                ('', True, False, False): self.bds.LIGHT_LEFT+' ',
                ('', False, False, True): self.bds.NO_VLINE,
                ('|', True, True, True): line_char+self.bds.LIGHT_CROSS+line_char,
                ('|', True, True, False): self.bds.LIGHT_LEFT+self.bds.LIGHT_VLINE.strip()+self.bds.LIGHT_RIGHT,
                ('|', False, True, False): ' '+self.bds.LIGHT_VERT_RIGHT+line_char,
                ('|', True, False, False): line_char+self.bds.LIGHT_VERT_LEFT+' ',
                ('|', False, False, True): self.bds.LIGHT_VLINE,
                ('||', True, True, True): line_char+self.bds.HEAVY_CROSS+line_char,
                ('||', True, True, False): self.bds.LIGHT_LEFT+self.bds.HEAVY_VLINE.strip()+self.bds.LIGHT_RIGHT,
                ('||', False, True, False): ' '+self.bds.HEAVY_VERT_RIGHT+line_char,
                ('||', True, False, False): line_char+self.bds.HEAVY_VERT_LEFT+' ',
                ('||', False, False, True): self.bds.HEAVY_VLINE,
                }

            def col_line_segment(col):
                """Helper returning the horizontal line segment for the column if any"""
                if horizontal_lines == True:
                    return (0,self._values.shape[1])
                for colrange in horizontal_lines:
                    if col in range(colrange[0], colrange[1]+1):
                        return colrange
                return None
                
            s = ''
            for col in range(self._values.shape[1]):
                cell_width = column_widths[col]

                vertical_line_type = self._get_vertical_line(col-1)
                vertical_line = self.ltype2str[vertical_line_type]
                if row == -1 or row == self._values.shape[0]-1:
                    s += len(vertical_line)*line_char
                elif vertical_line_type == '':
                    if col == 0:
                        s += self.bds.NO_VLINE
                    else:
                        key = (vertical_line_type,col_line_segment(col-1) is not None, col_line_segment(col) is not None,col_line_segment(col-1)==col_line_segment(col))
                        s += crossings[key]
                else:
                    key = (vertical_line_type,col_line_segment(col-1) is not None, col_line_segment(col) is not None,col_line_segment(col-1)==col_line_segment(col))
                    s += crossings[key]
                
                if col_line_segment(col):
                    s += cell_width * line_char
                else:
                    s += cell_width * ' '


            vertical_line_type = self._get_vertical_line(col)
            vertical_line = {'': self.bds.NO_VLINE, '|': self.bds.LIGHT_VLINE, '||': self.bds.HEAVY_VLINE}[vertical_line_type]
            if row == -1 or row == self._values.shape[0]-1:
                s += len(vertical_line)*line_char
            elif vertical_line_type == '':
                s += self.bds.NO_VLINE
            else:
                key = (vertical_line_type,col_line_segment(col-1) is not None, col_line_segment(col) is not None,col_line_segment(col-1)==col_line_segment(col))
                s += crossings[key]

            return s

        horizontal_line = horizontal_line_str(-1)
        if horizontal_line:
            lines.append(horizontal_line)

        for row in range(self._values.shape[0]):

            s = ''
            for col in range(self._values.shape[1]):
                value_cell_idx = value_cell_map[row,col]
                if value_cell_idx[1] != col:
                    continue
                col_format = self._get_column_format(col)
                vertical_line_type = self._get_vertical_line(col-1)
                vertical_line = self.ltype2str[vertical_line_type]
                s += vertical_line
                
                if value_cell_idx != (row, col):
                    cell = self._get_cell(*value_cell_idx)
                else:
                    cell = self._get_cell(row, col)
                cell_width = compute_cell_width(cell, col)

                attrs = {}
                if col_format.auto_highlight in ('largest', 'smallest') and extreme_values['cols'][col]:
                    attrs = extreme_values['cols'][col][col_format.auto_highlight].get(cell.number_after_format(self, row, col), {})
                
                s += cell.to_str(self, row, col, cell_width, line=row-value_cell_idx[0], **attrs)

            vertical_line_type = self._get_vertical_line(col)
            vertical_line = self.ltype2str[vertical_line_type]
            s += vertical_line

            lines.append(s)

            horizontal_line = horizontal_line_str(row)
            if horizontal_line:
                lines.append(horizontal_line)


        return '\n'.join(lines)

    def str(self):
        """Convert to a string that can be printed in a terminal."""
        return str(self)

    def latex(self, table_env: bool=False, document=False) -> str:
        """Convert to a latex booktabs table.
        
        Args:
            table_env: If True wrap the table inside a table environment.
            document: If True create a standalone latex document.
        Returns:
            Latex code for a booktabs table.
        """
        self._target = 'latex'
        lines = []
        indent_width = 4
        indent_level = 0
        self._check()

        extreme_values = self._extreme_values()
        value_cell_map = self._value_cell_map()

        def _indent():
            return indent_level*indent_width*' '

        def add_line(s):
            lines.append(_indent()+s)

        def num_format_to_table_format(num_format, max_value):
            match = re.match('{:[<^>]{0,1}(\d*)\.(\d+)f}', num_format)
            if match:
                width = len(num_format.format(max_value))
                _, decimals = match.groups()
                decimals = int(decimals)
                table_format = f'{width-decimals-1}.{decimals}'
                return table_format

        verts_map = {'': '', '|': '|', '||': '||'}

        if document:
            header = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{etoolbox}
\usepackage{siunitx}
\usepackage{multirow}
\usepackage[normalem]{ulem}

\begin{document}

"""
            for x in header.splitlines():
                add_line(x)

        if table_env:
            add_line(r'\begin{table}')
            indent_level += 1
            add_line(r'\robustify\bfseries')
            add_line(r'\robustify\uline')
            add_line(r'\centering')

        column_format_str = ''
        for col in range(self._values.shape[1]):
            col_format = self._get_column_format(col)
            if col == 0:
                column_format_str += verts_map[col_format.leftmost_line]

            if extreme_values['cols'][col]:
                table_format = num_format_to_table_format(col_format.num_format, max(extreme_values['cols'][col]['largest'].keys()))
            else:
                table_format = None
            if table_format:
                column_format_str += f'S[table-format={table_format},detect-all]'
            else:
                column_format_str += col_format.align 
            column_format_str += verts_map[col_format.line]
        add_line(r'\begin{tabular}{'+column_format_str+'}')
        if self._get_horizontal_line(-1):
            add_line(r'\toprule')

        # create table with all strings
        latex_str_tab = np.empty_like(self._values)
        for row in range(self._values.shape[0]):
            for col in range(self._values.shape[1]):
                value_cell_idx = value_cell_map[row,col]
                if value_cell_idx[1] != col:
                    continue
                col_format = self._get_column_format(col)
                
                if value_cell_idx != (row, col):
                    cell = self._get_cell(*value_cell_idx)
                    if cell.row_span > 1 and cell.col_span == 1:
                        latex_str_tab[row,col] = ''
                    elif cell.row_span > 1 and cell.col_span > 1:
                        latex_str_tab[row,col] = f'\\multicolumn{{{cell.col_span}}}{{c}}{{}}'
                    continue
                else:
                    cell = self._get_cell(row, col)
                
                attrs = {}
                if col_format.auto_highlight in ('largest', 'smallest') and extreme_values['cols'][col]:
                    attrs = extreme_values['cols'][col][col_format.auto_highlight].get(cell.number_after_format(self, row, col), {})

                align = 'c'
                if col_format and col_format.align:
                    align = col_format.align

                if cell.row_span == 1 and cell.col_span == 1:
                    latex_str_tab[row,col] = cell.to_latex(self, row, col, **attrs)
                elif cell.row_span == 1 and cell.col_span > 1:
                    latex_str_tab[row,col] = f'\\multicolumn{{{cell.col_span}}}{{{align}}}{{{cell.to_latex(self, row, col, **attrs)}}}'
                elif cell.row_span > 1 and cell.col_span > 1:
                    latex_str_tab[row,col] = f'\\multicolumn{{{cell.col_span}}}{{{align}}}{{\\multirow{{{cell.row_span}}}{{*}}{{{cell.to_latex(self, row, col, **attrs)}}}}}'


        column_widths = []
        for col in range(latex_str_tab.shape[1]):
            widths = []
            for row in range(latex_str_tab.shape[0]):
                cell = self._get_cell(row,col)
                if isinstance(latex_str_tab[row,col], str) and cell.col_span == 1:
                    widths.append(len(latex_str_tab[row,col]))
            if widths:
                column_widths.append(max(widths))
            else:
                column_widths.append(4)
        for col in range(latex_str_tab.shape[1]):
            for row in range(latex_str_tab.shape[0]):
                cell = self._get_cell(row,col)
                if isinstance(latex_str_tab[row,col], str) and cell.col_span >= 1:
                    w = len(latex_str_tab[row,col])
                    col_span_w = sum(column_widths[col:col+cell.col_span])
                    if w > col_span_w:
                        column_widths[col] += w - col_span_w

        def compute_cell_width(cell, col):
            w = sum(column_widths[col:col+cell.col_span])
            w += 3 * (cell.col_span-1)
            return w

        for row in range(latex_str_tab.shape[0]):
            strings = []
            for col in range(latex_str_tab.shape[1]):
                cell_str = latex_str_tab[row,col]
                if isinstance(cell_str, str):
                    width = compute_cell_width(self._get_cell(*value_cell_map[row,col]), col)
                    strings.append(cell_str.ljust(width))

            s = ' & '.join(strings) + r' \\'
            horizontal_line = self._get_horizontal_line(row)
            if horizontal_line and row < latex_str_tab.shape[0]-1:
                if isinstance(horizontal_line, bool):
                    s += ' \midrule'
                elif isinstance(horizontal_line,(list,tuple)):
                    for colrange in horizontal_line:
                        s+= f' \cmidrule(lr){{{colrange[0]+1}-{colrange[1]+1}}}'
                else:
                    raise Exception(f'unsupported value {type(horizontal_line)}')

            add_line(s)
            if horizontal_line and row == latex_str_tab.shape[0]-1:
                add_line(r'\bottomrule')

        add_line(r'\end{tabular}')

        if table_env:
            indent_level -= 1
            add_line(r'\end{table}')

        if document:
            add_line('')
            add_line(r'\end{document}')

        return '\n'.join(lines)