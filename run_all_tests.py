#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE TEST RUNNER - Reddit Mention Tracker
====================================================

This script runs ALL tests in the correct order with comprehensive validation.
Use this to test the entire system at once.

Usage:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py --quick           # Run only critical tests (5 min)
    python run_all_tests.py --no-performance  # Skip performance tests
    python run_all_tests.py --verbose         # Detailed output
"""

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse


class ComprehensiveTestRunner:
    """Runs all tests with comprehensive validation and reporting."""
    
    def __init__(self, project_root: str = None):
        """Initialize the comprehensive test runner."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.start_time = time.time()
        self.results = {}
        self.total_tests = 0
        self.total_passed = 0
        self.total_failed = 0
        
        # Ensure we're in the right directory
        os.chdir(self.project_root)
        
        print("*** REDDIT MENTION TRACKER - COMPREHENSIVE TEST SUITE ***")
        print("=" * 60)
        print(f"Project Root: {self.project_root}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("\n*** CHECKING PREREQUISITES...")
        
        # Check Python version
        if sys.version_info < (3, 11):
            print("[X] Python 3.11+ required")
            return False
        print(f"[OK] Python {sys.version.split()[0]}")
        
        # Check required directories
        required_dirs = ['tests', 'database', 'scraper', 'analytics', 'ui', 'api']
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                print(f"[X] Missing directory: {dir_name}")
                return False
        print("[OK] All required directories present")
        
        # Check test files
        test_files = [
            'tests/test_database.py',
            'tests/test_scraper.py', 
            'tests/test_analytics.py',
            'tests/test_ui.py',
            'tests/test_integration.py',
            'tests/test_runner.py'
        ]
        
        for test_file in test_files:
            if not (self.project_root / test_file).exists():
                print(f"[X] Missing test file: {test_file}")
                return False
        print("[OK] All test files present")
        
        # Check if pytest is available
        try:
            subprocess.run([sys.executable, '-m', 'pytest', '--version'], 
                         check=True, capture_output=True)
            print("[OK] Pytest available")
        except subprocess.CalledProcessError:
            print("[X] Pytest not available")
            return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install required test dependencies."""
        print("\n*** INSTALLING TEST DEPENDENCIES...")
        
        dependencies = [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.0.0',
            'pytest-html>=3.1.0',
            'pytest-xdist>=3.0.0',
            'coverage>=7.0.0',
            'mock>=4.0.0',
            'psutil>=5.9.0'
        ]
        
        for dep in dependencies:
            try:
                print(f"Installing {dep}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"[Warning] Could not install {dep}")
        
        print("[OK] Dependencies installation completed")
        return True
    
    def run_test_category(self, category: str, description: str, 
                         timeout: int = 300, critical: bool = True) -> Dict:
        """Run a specific test category."""
        print(f"\n*** {description.upper()}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Use pytest directly with markers
            cmd = [sys.executable, '-m', 'pytest', 'tests/', '-m', category, '-v', '--tb=short']
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=timeout, cwd=self.project_root)
            
            execution_time = time.time() - start_time
            
            # Parse results from pytest output
            success = result.returncode == 0
            output_lines = result.stdout.split('\n')
            
            # Extract statistics from pytest output
            stats = self._extract_pytest_stats(output_lines)
            
            test_result = {
                'category': category,
                'description': description,
                'success': success,
                'execution_time': execution_time,
                'statistics': stats,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'critical': critical
            }
            
            # Update totals
            self.total_tests += stats.get('total', 0)
            self.total_passed += stats.get('passed', 0)
            self.total_failed += stats.get('failed', 0)
            
            # Print summary
            if success:
                print(f"[PASS] {description}: PASSED")
                print(f"   Stats: {stats.get('total', 0)} | "
                      f"Passed: {stats.get('passed', 0)} | "
                      f"Failed: {stats.get('failed', 0)}")
                print(f"   Time: {execution_time:.1f}s")
            else:
                print(f"[FAIL] {description}: FAILED")
                print(f"   Stats: {stats.get('total', 0)} | "
                      f"Passed: {stats.get('passed', 0)} | "
                      f"Failed: {stats.get('failed', 0)}")
                print(f"   Time: {execution_time:.1f}s")
                if critical:
                    print(f"   [CRITICAL] This is a required component")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {description}: TIMEOUT after {timeout}s")
            return {
                'category': category,
                'description': description,
                'success': False,
                'execution_time': timeout,
                'error': 'timeout',
                'critical': critical
            }
        except Exception as e:
            print(f"[ERROR] {description}: ERROR - {e}")
            return {
                'category': category,
                'description': description,
                'success': False,
                'execution_time': 0,
                'error': str(e),
                'critical': critical
            }
    
    def _extract_pytest_stats(self, output_lines: List[str]) -> Dict:
        """Extract test statistics from pytest output."""
        stats = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0}
        
        for line in output_lines:
            line = line.strip()
            
            # Look for pytest summary line like "=============== 7 passed, 168 deselected, 115 warnings in 9.64s ==============="
            if ('passed' in line or 'failed' in line) and ' in ' in line and '=' in line:
                # Remove the equals signs and extract the middle part
                middle_part = line.replace('=', '').strip()
                
                # Split by commas and parse each part
                parts = middle_part.split(' in ')[0].split(',')  # Remove timing part
                
                for part in parts:
                    part = part.strip()
                    if ' passed' in part:
                        try:
                            stats['passed'] = int(part.split()[0])
                        except (ValueError, IndexError):
                            pass
                    elif ' failed' in part:
                        try:
                            stats['failed'] = int(part.split()[0])
                        except (ValueError, IndexError):
                            pass
                    elif ' skipped' in part:
                        try:
                            stats['skipped'] = int(part.split()[0])
                        except (ValueError, IndexError):
                            pass
                    elif ' error' in part:
                        try:
                            stats['errors'] = int(part.split()[0])
                        except (ValueError, IndexError):
                            pass
                
                # Found the summary line, break
                break
        
        stats['total'] = stats['passed'] + stats['failed'] + stats['skipped'] + stats['errors']
        return stats
    
    def run_smoke_tests(self) -> bool:
        """Run smoke tests for quick validation."""
        print("💨 SMOKE TESTS - Quick System Validation")
        print("-" * 50)
        
        try:
            # Run smoke tests using pytest directly
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'tests/', '-m', 'smoke', '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)
            
            # Check if smoke tests passed
            if result.returncode == 0:
                # Parse output to get actual test results
                output_lines = result.stdout.split('\n')
                stats = self._extract_pytest_stats(output_lines)
                
                # Check if we have successful test results
                if stats['passed'] > 0 and stats['failed'] == 0:
                    print(f"[PASS] Smoke Tests: PASSED ({stats['passed']} tests passed)")
                    return True
                elif stats['passed'] == 0 and stats['failed'] == 0:
                    print("[FAIL] Smoke Tests: No tests executed")
                    return False
                else:
                    print(f"[FAIL] Smoke Tests: FAILED ({stats['failed']} failed, {stats['passed']} passed)")
                    return False
            else:
                print("[FAIL] Smoke Tests: FAILED - Test runner returned error")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                return False
                
        except subprocess.TimeoutExpired:
            print("[TIMEOUT] Smoke Tests: TIMEOUT - Tests took too long")
            return False
        except Exception as e:
            print(f"[ERROR] Smoke Tests: ERROR - {str(e)}")
            return False
    
    def run_all_tests(self, quick_mode: bool = False, 
                     skip_performance: bool = False, 
                     verbose: bool = False) -> Dict:
        """Run all test categories in sequence."""
        
        # Step 1: Smoke tests (critical)
        smoke_result = self.run_smoke_tests()
        if not smoke_result:
            return {'success': False, 'results': {'smoke': {'success': False, 'critical_failure': True}}}
        
        self.results['smoke'] = {'success': smoke_result, 'critical_failure': False}
        
        if quick_mode:
            print("\n*** QUICK MODE - Running essential tests only")
            test_sequence = [
                ('unit', 'Unit Tests - Core Components', 300, True),
                ('integration', 'Integration Tests - Component Interaction', 600, True)
            ]
        else:
            print("\n*** FULL MODE - Running comprehensive test suite")
            test_sequence = [
                ('unit', 'Unit Tests - Core Components', 600, True),
                ('integration', 'Integration Tests - Component Interaction', 900, True),
                ('security', 'Security Tests - Vulnerability Assessment', 300, True),
            ]
            
            if not skip_performance:
                test_sequence.append(
                    ('performance', 'Performance Tests - Load & Stress Testing', 1200, False)
                )
        
        # Run test sequence
        critical_failure = False
        for category, description, timeout, critical in test_sequence:
            result = self.run_test_category(category, description, timeout, critical)
            self.results[category] = result
            
            if not result['success'] and critical:
                critical_failure = True
                print(f"\n[CRITICAL] CRITICAL FAILURE in {description}")
                print("   Consider fixing this before continuing...")
        
        return {
            'success': not critical_failure,
            'results': self.results,
            'critical_failure': critical_failure
        }
    
    def run_manual_validation(self) -> Dict:
        """Run manual validation checks."""
        print("\n*** MANUAL VALIDATION CHECKS")
        print("-" * 50)
        
        validations = []
        
        # Check if main files exist and are importable
        core_modules = [
            'database.models',
            'scraper.reddit_scraper', 
            'analytics.metrics_analyzer',
            'ui.visualization',
            'app'
        ]
        
        for module in core_modules:
            try:
                __import__(module)
                print(f"[OK] {module}: Importable")
                validations.append({'module': module, 'status': 'pass'})
            except Exception as e:
                print(f"[X] {module}: Import failed - {e}")
                validations.append({'module': module, 'status': 'fail', 'error': str(e)})
        
        # Check database creation
        try:
            from database.models import DatabaseManager
            db = DatabaseManager("sqlite:///validation_test.db")
            db.create_tables()
            db.close()
            os.remove("validation_test.db")
            print("[OK] Database: Creation successful")
            validations.append({'component': 'database', 'status': 'pass'})
        except Exception as e:
            print(f"[X] Database: Creation failed - {e}")
            validations.append({'component': 'database', 'status': 'fail', 'error': str(e)})
        
        # Check if requirements.txt exists
        if (self.project_root / 'requirements.txt').exists():
            print("[OK] Requirements: File exists")
            validations.append({'component': 'requirements', 'status': 'pass'})
        else:
            print("[X] Requirements: File missing")
            validations.append({'component': 'requirements', 'status': 'fail'})
        
        success = all(v.get('status') == 'pass' for v in validations)
        return {'success': success, 'validations': validations}
    
    def generate_final_report(self, test_results: Dict) -> None:
        """Generate comprehensive final report."""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("*** COMPREHENSIVE TEST SUITE - FINAL REPORT ***")
        print("=" * 80)
        
        print(f"Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Total Tests Executed: {self.total_tests}")
        print(f"Total Passed: {self.total_passed}")
        print(f"Total Failed: {self.total_failed}")
        
        if self.total_tests > 0:
            success_rate = (self.total_passed / self.total_tests) * 100
            print(f"Overall Success Rate: {success_rate:.1f}%")
        else:
            success_rate = 0
            print("Overall Success Rate: N/A (No tests executed)")
        
        print("\n*** COMPONENT STATUS:")
        print("-" * 40)
        
        for category, result in test_results.get('results', {}).items():
            if category == 'smoke':
                continue
                
            status = "[PASS]" if result.get('success') else "[FAIL]"
            critical = "[CRITICAL]" if result.get('critical') else "[OPTIONAL]"
            time_taken = result.get('execution_time', 0)
            
            print(f"{status} | {critical} | {category.upper():12} | {time_taken:6.1f}s")
        
        print("\n*** OVERALL SYSTEM STATUS:")
        print("-" * 40)
        
        if test_results.get('success') and success_rate >= 95:
            print("[EXCELLENT] System is production ready!")
            print("   All critical components working perfectly")
        elif test_results.get('success') and success_rate >= 85:
            print("[GOOD] System is functional with minor issues")
            print("   Consider addressing failed tests for optimal performance")
        elif test_results.get('success') and success_rate >= 70:
            print("[ACCEPTABLE] System works but needs improvement")
            print("   Several components need attention")
        else:
            print("[CRITICAL] System needs immediate attention")
            print("   Fix critical failures before deployment")
        
        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': total_time,
            'total_tests': self.total_tests,
            'total_passed': self.total_passed,
            'total_failed': self.total_failed,
            'success_rate': success_rate,
            'overall_success': test_results.get('success'),
            'results': test_results.get('results', {}),
            'system_status': 'production_ready' if success_rate >= 95 else 'needs_attention'
        }
        
        # Create reports directory if it doesn't exist
        reports_dir = self.project_root / 'test_reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f'comprehensive_test_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\nDetailed report saved: {report_file}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for Reddit Mention Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tests.py                    # Run all tests (30-45 minutes)
  python run_all_tests.py --quick           # Run essential tests only (10-15 minutes)
  python run_all_tests.py --no-performance  # Skip performance tests
  python run_all_tests.py --verbose         # Detailed output
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run only essential tests (faster)')
    parser.add_argument('--no-performance', action='store_true',
                       help='Skip performance tests')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--project-root', 
                       help='Project root directory (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        # Initialize test runner
        runner = ComprehensiveTestRunner(args.project_root)
        
        # Check prerequisites
        if not runner.check_prerequisites():
            print("\n[ERROR] Prerequisites check failed!")
            print("Please ensure all required files and dependencies are present.")
            sys.exit(1)
        
        # Install dependencies
        runner.install_dependencies()
        
        # Run manual validation
        validation_result = runner.run_manual_validation()
        if not validation_result['success']:
            print("\n[WARNING] Manual validation found issues, but continuing with tests...")
        
        # Run all tests
        print(f"\n*** Starting comprehensive test execution...")
        if args.quick:
            print("   Mode: QUICK (essential tests only)")
        else:
            print("   Mode: FULL (comprehensive testing)")
        
        test_results = runner.run_all_tests(
            quick_mode=args.quick,
            skip_performance=args.no_performance,
            verbose=args.verbose
        )
        
        # Generate final report
        runner.generate_final_report(test_results)
        
        # Exit with appropriate code
        if test_results.get('success'):
            print("\n[SUCCESS] All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n[FAIL] Some tests failed. Check the report for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[WARNING] Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 