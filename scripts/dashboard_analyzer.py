#!/usr/bin/env python3
"""
Dashboard Analysis Script
Analyzes the current dashboard structure and provides refactoring insights
"""
import ast
import os
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import argparse

class DashboardAnalyzer:
    """Analyze dashboard structure for refactoring insights"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.tree = None
        self.functions = []
        self.classes = []
        self.imports = []
        self.globals = []
        
    def load_and_parse(self) -> None:
        """Load and parse the Python file"""
        with open(self.file_path, 'r') as file:
            content = file.read()
        self.tree = ast.parse(content)
        
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze the overall structure of the dashboard"""
        if not self.tree:
            self.load_and_parse()
            
        analysis = {
            'file_info': self._get_file_info(),
            'functions': self._analyze_functions(),
            'classes': self._analyze_classes(),
            'imports': self._analyze_imports(),
            'complexity': self._analyze_complexity(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _get_file_info(self) -> Dict[str, Any]:
        """Get basic file information"""
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            
        return {
            'total_lines': len(lines),
            'non_empty_lines': sum(1 for line in lines if line.strip()),
            'comment_lines': sum(1 for line in lines if line.strip().startswith('#')),
            'docstring_lines': self._count_docstring_lines(),
            'file_size_kb': os.path.getsize(self.file_path) / 1024
        }
    
    def _count_docstring_lines(self) -> int:
        """Count lines in docstrings"""
        count = 0
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if (ast.get_docstring(node)):
                    docstring = ast.get_docstring(node)
                    count += len(docstring.split('\n'))
        return count
    
    def _analyze_functions(self) -> List[Dict[str, Any]]:
        """Analyze all functions in the file"""
        functions = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                func_analysis = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno if hasattr(node, 'end_lineno') else None,
                    'num_lines': (node.end_lineno - node.lineno + 1) if hasattr(node, 'end_lineno') else None,
                    'num_args': len(node.args.args),
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'complexity_score': self._calculate_function_complexity(node),
                    'calls_streamlit': self._check_streamlit_usage(node),
                    'has_docstring': bool(ast.get_docstring(node)),
                    'category': self._categorize_function(node.name)
                }
                functions.append(func_analysis)
        
        # Sort by line count (descending) for refactoring priority
        return sorted(functions, key=lambda x: x.get('num_lines', 0), reverse=True)
    
    def _analyze_classes(self) -> List[Dict[str, Any]]:
        """Analyze all classes in the file"""
        classes = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_analysis = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno if hasattr(node, 'end_lineno') else None,
                    'num_methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    'inheritance': [base.id for base in node.bases if isinstance(base, ast.Name)],
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list]
                }
                classes.append(class_analysis)
        
        return classes
    
    def _analyze_imports(self) -> Dict[str, List[str]]:
        """Analyze imports to understand dependencies"""
        imports = defaultdict(list)
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports['standard'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    category = self._categorize_import(module)
                    imports[category].append(f"{module}.{alias.name}")
        
        return dict(imports)
    
    def _analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity metrics"""
        total_functions = len([n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)])
        total_classes = len([n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)])
        
        # Calculate cyclomatic complexity
        complexity_scores = [
            self._calculate_function_complexity(node) 
            for node in ast.walk(self.tree) 
            if isinstance(node, ast.FunctionDef)
        ]
        
        return {
            'total_functions': total_functions,
            'total_classes': total_classes,
            'avg_complexity': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            'max_complexity': max(complexity_scores) if complexity_scores else 0,
            'high_complexity_functions': [
                func['name'] for func in self._analyze_functions() 
                if func['complexity_score'] > 10
            ]
        }
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
                
        return complexity
    
    def _get_decorator_name(self, decorator) -> str:
        """Get decorator name as string"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{decorator.value.id}.{decorator.attr}" if isinstance(decorator.value, ast.Name) else decorator.attr
        else:
            return str(decorator)
    
    def _check_streamlit_usage(self, node: ast.FunctionDef) -> bool:
        """Check if function uses Streamlit components"""
        streamlit_keywords = ['st.', 'streamlit.', 'plotly', 'chart', 'graph']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if hasattr(child.value, 'id') and child.value.id in ['st', 'streamlit']:
                    return True
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    func_name = ast.unparse(child.func) if hasattr(ast, 'unparse') else str(child.func)
                    if any(keyword in func_name for keyword in streamlit_keywords):
                        return True
        
        return False
    
    def _categorize_function(self, func_name: str) -> str:
        """Categorize function by its purpose"""
        name_lower = func_name.lower()
        
        if any(keyword in name_lower for keyword in ['chart', 'plot', 'graph', 'visual']):
            return 'visualization'
        elif any(keyword in name_lower for keyword in ['signal', 'indicator', 'technical']):
            return 'signal_processing'
        elif any(keyword in name_lower for keyword in ['data', 'load', 'fetch', 'get']):
            return 'data_management'
        elif any(keyword in name_lower for keyword in ['risk', 'portfolio', 'position']):
            return 'risk_management'
        elif any(keyword in name_lower for keyword in ['style', 'format', 'display', 'render']):
            return 'ui_presentation'
        else:
            return 'utility'
    
    def _categorize_import(self, module: str) -> str:
        """Categorize imports by their purpose"""
        if not module:
            return 'builtin'
        
        if module.startswith('streamlit') or module.startswith('st'):
            return 'streamlit'
        elif module.startswith('plotly') or module.startswith('matplotlib'):
            return 'visualization'
        elif module.startswith('pandas') or module.startswith('numpy'):
            return 'data_science'
        elif module.startswith('src.'):
            return 'internal'
        else:
            return 'external'
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate refactoring recommendations"""
        recommendations = []
        functions = self._analyze_functions()
        complexity = self._analyze_complexity()
        
        # Large function recommendations
        large_functions = [f for f in functions if f.get('num_lines', 0) > 50]
        if large_functions:
            recommendations.append({
                'type': 'refactor_large_functions',
                'priority': 'high',
                'description': f"Found {len(large_functions)} functions with >50 lines. Consider breaking them down.",
                'functions': [f['name'] for f in large_functions[:5]],
                'estimated_effort': len(large_functions) * 2  # hours
            })
        
        # High complexity recommendations
        if complexity['high_complexity_functions']:
            recommendations.append({
                'type': 'reduce_complexity',
                'priority': 'medium',
                'description': f"Found {len(complexity['high_complexity_functions'])} high-complexity functions.",
                'functions': complexity['high_complexity_functions'][:5],
                'estimated_effort': len(complexity['high_complexity_functions']) * 1.5
            })
        
        # Component separation recommendations
        function_categories = defaultdict(list)
        for func in functions:
            function_categories[func['category']].append(func['name'])
        
        for category, func_list in function_categories.items():
            if len(func_list) > 5:
                recommendations.append({
                    'type': 'component_separation',
                    'priority': 'high',
                    'description': f"Consider creating a separate {category} component with {len(func_list)} functions.",
                    'category': category,
                    'functions': func_list[:10],
                    'estimated_effort': 8  # hours per component
                })
        
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        analysis = self.analyze_structure()
        
        # Add summary statistics
        analysis['summary'] = {
            'refactoring_priority': self._calculate_refactoring_priority(analysis),
            'estimated_effort_hours': self._estimate_refactoring_effort(analysis),
            'component_suggestions': self._suggest_components(analysis)
        }
        
        return analysis
    
    def _calculate_refactoring_priority(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall refactoring priority"""
        file_size = analysis['file_info']['file_size_kb']
        avg_complexity = analysis['complexity']['avg_complexity']
        large_functions = len([f for f in analysis['functions'] if f.get('num_lines', 0) > 50])
        
        score = 0
        if file_size > 100:  # >100KB file
            score += 3
        if avg_complexity > 8:
            score += 2
        if large_functions > 10:
            score += 2
        
        if score >= 5:
            return 'critical'
        elif score >= 3:
            return 'high'
        elif score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_refactoring_effort(self, analysis: Dict[str, Any]) -> int:
        """Estimate total refactoring effort in hours"""
        base_effort = 8  # Base refactoring effort
        
        # Add effort for each large function
        large_functions = len([f for f in analysis['functions'] if f.get('num_lines', 0) > 50])
        base_effort += large_functions * 2
        
        # Add effort for each component
        recommendations = [r for r in analysis['recommendations'] if r['type'] == 'component_separation']
        base_effort += len(recommendations) * 8
        
        return base_effort
    
    def _suggest_components(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest component breakdown"""
        function_categories = defaultdict(list)
        for func in analysis['functions']:
            function_categories[func['category']].append(func['name'])
        
        components = []
        for category, functions in function_categories.items():
            if len(functions) > 3:
                components.append(f"{category}_component")
        
        return components

def main():
    parser = argparse.ArgumentParser(description='Analyze dashboard structure for refactoring')
    parser.add_argument('file_path', help='Path to the dashboard file to analyze')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} not found")
        return 1
    
    analyzer = DashboardAnalyzer(args.file_path)
    report = analyzer.generate_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Analysis report saved to {args.output}")
    
    if args.verbose:
        print(json.dumps(report, indent=2, default=str))
    else:
        # Print summary
        print("=== Dashboard Analysis Summary ===")
        print(f"File size: {report['file_info']['file_size_kb']:.1f} KB")
        print(f"Total lines: {report['file_info']['total_lines']}")
        print(f"Functions: {report['complexity']['total_functions']}")
        print(f"Average complexity: {report['complexity']['avg_complexity']:.1f}")
        print(f"Refactoring priority: {report['summary']['refactoring_priority'].upper()}")
        print(f"Estimated effort: {report['summary']['estimated_effort_hours']} hours")
        print(f"Suggested components: {', '.join(report['summary']['component_suggestions'])}")
        
        print("\n=== Top Recommendations ===")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"{i}. {rec['description']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    main()